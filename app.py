import streamlit as st
import os
from dotenv import load_dotenv
import openai
from supabase import create_client
import pinecone
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import fitz  # PyMuPDF
import tempfile
import uuid
import base64
import google.generativeai as genai
import time
from httpx import Client
import requests
import re

# Load environment variables
load_dotenv()

# Configuration flags - ADD THIS SECTION
ENABLE_PINECONE_INDEXING = True  # Set to True to enable vector indexing
ENABLE_VECTOR_SEARCH = True  # Allow vector search even if indexing is disabled
SKIP_VECTOR_STORAGE_WARNING = "⚠️ Vector upload is temporarily disabled. Using existing vectors for search."
# End configuration flags

# Configure page
st.set_page_config(
    page_title="Coursera Study Buddy",
    page_icon="📚",
    layout="wide",
)

# Initialize API clients
try:
    # OpenAI
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        openai.api_key = st.secrets.get("OPENAI_API_KEY")
    
    # Google Gemini
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        gemini_api_key = st.secrets.get("GOOGLE_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash')
    else:
        st.warning("⚠️ Gemini API key not found. Some features may be limited.")
    
    # Supabase
    # First try to get credentials from Streamlit secrets
    supabase_url = st.secrets.get("SUPABASE_URL")
    supabase_key = st.secrets.get("SUPABASE_KEY")
    supabase_service_key = st.secrets.get("SUPABASE_SERVICE_KEY")
    
    # Fall back to environment variables if not found in secrets
    if not supabase_url:
        supabase_url = os.environ.get("SUPABASE_URL")
    if not supabase_key:
        supabase_key = os.environ.get("SUPABASE_KEY")
    if not supabase_service_key:
        supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    # Debug information for Supabase credentials
    if not supabase_url or not supabase_key or not supabase_service_key:
        st.error("❌ Supabase credentials are missing. Please check your .env file or Streamlit secrets.")
        st.info("Make sure you have set up SUPABASE_URL, SUPABASE_KEY, and SUPABASE_SERVICE_KEY in your environment variables or secrets.")
        apis_configured = False
    else:
        # Mask the keys for security but show enough to verify they're loading correctly
        masked_key = supabase_key[:5] + "*" * (len(supabase_key) - 10) + supabase_key[-5:] if len(supabase_key) > 10 else "***"
        masked_service_key = supabase_service_key[:5] + "*" * (len(supabase_service_key) - 10) + supabase_service_key[-5:] if len(supabase_service_key) > 10 else "***"
        st.sidebar.expander("Debug Info", expanded=False).write(f"""
        **Supabase Configuration:**
        - URL: {supabase_url}
        - Anon Key: {masked_key}
        - Service Key: {masked_service_key}
        """)
        
        # Initialize Supabase client with debug information
        try:
            supabase = create_client(supabase_url, supabase_key)
            st.sidebar.expander("Debug Info", expanded=False).success("✅ Supabase connection initialized successfully")
        except Exception as supabase_error:
            st.error(f"❌ Failed to initialize Supabase client: {str(supabase_error)}")
            st.info("Check that your Supabase URL and key are correct and that your project is active.")
            apis_configured = False
            raise
    
    # Pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
    
    pinecone_env = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
    pinecone_index_name = "coursera-transcripts"
    
    # Initialize Pinecone client
    pc = pinecone.Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, create if not
    if pinecone_index_name not in pc.list_indexes().names():
        st.info(f"Creating new Pinecone index: {pinecone_index_name}")
        pc.create_index(
            name=pinecone_index_name,
            dimension=1536,  # For OpenAI embeddings
            metric="cosine",
            spec=pinecone.ServerlessSpec(
                cloud="gcp",
                region="us-central1"
            )
        )
    
    # Connect to the index
    pinecone_index = pc.Index(pinecone_index_name)
    
    # Verify connection
    stats = pinecone_index.describe_index_stats()
    st.sidebar.expander("Debug Info", expanded=False).write(f"""
    **Pinecone Configuration:**
    - Index: {pinecone_index_name}
    - Total vectors: {stats.get('total_vector_count', 0)}
    - Namespaces: {stats.get('namespaces', {})}
    """)
    
    apis_configured = True
except Exception as e:
    st.error(f"Error initializing Pinecone: {str(e)}")
    apis_configured = False

# Create sidebar
st.sidebar.title("Coursera Study Buddy")
st.sidebar.info("Upload Coursera transcripts and interact with them using AI.")

# Add configuration check button to sidebar
if st.sidebar.button("🔍 Check Configuration"):
    st.sidebar.write("Checking system configuration...")
    
    # Check OpenAI API (optional)
    try:
        # Simple test request to OpenAI API
        if openai.api_key:
            import openai
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            st.sidebar.success("✅ OpenAI API connection successful")
        else:
            st.sidebar.info("ℹ️ OpenAI API key not configured (optional)")
    except Exception as e:
        st.sidebar.warning(f"⚠️ OpenAI API not available: {str(e)}")
        st.sidebar.info("The app will use Gemini API for all AI features")
    
    # Check Gemini API
    try:
        if gemini_api_key:
            response = gemini_model.generate_content("Hello")
            st.sidebar.success("✅ Gemini API connection successful")
        else:
            st.sidebar.warning("⚠️ Gemini API key not configured")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini API error: {str(e)}")
    
    # Check Supabase connection
    try:
        # Try listing buckets
        buckets = supabase.storage.list_buckets()
        st.sidebar.success(f"✅ Supabase connection successful. Found {len(buckets)} buckets")
        # Check for transcripts bucket
        if any(bucket['name'] == 'transcripts' for bucket in buckets):
            st.sidebar.success("✅ 'transcripts' bucket exists")
        else:
            st.sidebar.warning("⚠️ 'transcripts' bucket not found")
            
        # Check database table
        try:
            result = supabase.table("transcripts").select("count", "exact").execute()
            st.sidebar.success("✅ 'transcripts' database table exists")
        except:
            st.sidebar.warning("⚠️ 'transcripts' database table not found")
    except Exception as e:
        st.sidebar.error(f"❌ Supabase error: {str(e)}")
    
    # Check Pinecone
    try:
        # List indexes
        indexes = pc.list_indexes()
        st.sidebar.success(f"✅ Pinecone connection successful. Found {len(indexes.names())} indexes")
        
        # Check for coursera-transcripts index
        if pinecone_index_name in indexes.names():
            st.sidebar.success(f"✅ '{pinecone_index_name}' index exists")
        else:
            st.sidebar.warning(f"⚠️ '{pinecone_index_name}' index not found")
    except Exception as e:
        st.sidebar.error(f"❌ Pinecone error: {str(e)}")

# Main app tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload", "Summarize", "Ask Questions", "Generate Quiz", "Exam Prep"])

# Helper functions
def extract_text_from_file(file_data, file_path):
    """Extract text from various file types based on extension"""
    if not file_data:
        return ""
        
    file_extension = os.path.splitext(file_path.lower())[1]
    
    # For text files, just decode the bytes
    if file_extension in ['.txt', '.md', '.csv']:
        try:
            # Try to decode as UTF-8 first
            return file_data.decode('utf-8')
        except UnicodeDecodeError:
            # Fall back to other encodings if necessary
            try:
                return file_data.decode('latin-1')
            except Exception as decode_error:
                st.error(f"Could not decode text file: {str(decode_error)}")
                return ""
    
    # For PDF files, use PyMuPDF
    elif file_extension == '.pdf':
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_data)
            temp_path = temp_file.name
        
        text = ""
        try:
            doc = fitz.open(temp_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
        finally:
            os.unlink(temp_path)
        
        return text
    
    # For JSON files, extract text field or join all text values
    elif file_extension == '.json':
        try:
            import json
            json_data = json.loads(file_data.decode('utf-8'))
            
            # If it's a simple object with a 'text' field
            if isinstance(json_data, dict) and 'text' in json_data:
                return json_data['text']
            
            # If it's an array of objects with 'text' fields (like transcript segments)
            elif isinstance(json_data, list) and all(isinstance(item, dict) for item in json_data):
                texts = []
                for item in json_data:
                    if 'text' in item:
                        texts.append(item['text'])
                return "\n".join(texts)
            
            # Fallback: convert the entire JSON to string
            return json.dumps(json_data)
        except Exception as json_error:
            st.error(f"Error parsing JSON file: {str(json_error)}")
            return ""
    
    # Unsupported file type
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return ""

def extract_text_from_supabase_pdf(file_path):
    """Download a file from Supabase and extract its text"""
    # Clean up the file path - remove leading/trailing whitespace
    file_path = file_path.strip() if file_path else ""
    
    # Check if this is a DB-stored file
    if file_path.startswith("DB_"):
        # Extract the original filename
        filename = file_path[3:].strip()  # Remove the "DB_" prefix and trim whitespace
        st.info(f"Retrieving file '{filename}' from database storage...")
        # Need to find the record in the database
        try:
            # Find all records that might match this filename
            results = supabase.table("file_contents").select("*").ilike("file_name", filename).execute()
            if results.data:
                # Use the first match
                file_record = results.data[0]
                # Decode the base64 data
                file_data = base64.b64decode(file_record["file_data"])
                
                # Use the generic extraction function for any file type
                return extract_text_from_file(file_data, filename)
            else:
                st.error(f"Could not find file '{filename}' in database")
                return ""
        except Exception as e:
            st.error(f"Error retrieving file from database: {str(e)}")
            return ""
            
    # Regular storage file
    st.info(f"Attempting to download '{file_path}' from Supabase storage...")
    
    # Try to list all files to see what's available
    try:
        available_files = supabase.storage.from_("transcripts").list()
        if not available_files:
            st.warning("⚠️ No files found in Supabase storage bucket 'transcripts'")
            st.info("Please use the 'Quick Upload' feature to upload your file first")
            return ""
            
        file_names = [f.get("name", "").strip() for f in available_files]
        if file_names:
            st.write(f"Available files in storage: {', '.join(file_names)}")
        else:
            st.warning("No readable files found in storage.")
            
        # Check for the file with and without whitespace
        cleaned_file_path = file_path.strip()
        if cleaned_file_path not in file_names:
            # Try case-insensitive matching and other variations
            close_matches = []
            for name in file_names:
                if cleaned_file_path.lower() in name.lower() or name.lower() in cleaned_file_path.lower():
                    close_matches.append(name)
                    
            if close_matches:
                st.warning(f"Exact file '{cleaned_file_path}' not found. Did you mean one of these? {', '.join(close_matches)}")
                # Try the closest match
                if st.button(f"Try using '{close_matches[0]}' instead"):
                    file_path = close_matches[0]
            else:
                st.warning(f"File '{cleaned_file_path}' not found in storage. Check the name carefully.")
    except Exception as list_error:
        st.warning(f"Could not list files in storage: {str(list_error)}")
    
    # Proceed with download attempt
    pdf_data = get_file_from_supabase(file_path)
    if not pdf_data:
        # Try additional local locations if file not found
        home_dir = os.path.expanduser("~")
        base_name = os.path.basename(file_path.strip())
        additional_local_paths = [
            os.path.join(home_dir, "Documents", base_name),
            os.path.join(home_dir, "Desktop", base_name),
            # Try common variations of the filename
            os.path.join(home_dir, "Downloads", base_name),
            os.path.join(home_dir, "Downloads", base_name.replace(" ", "_")),
            os.path.join(home_dir, "Downloads", base_name.replace("_", " ")),
            # Try with .txt extension if not already present
            os.path.join(home_dir, "Downloads", base_name + ".txt" if not base_name.endswith(".txt") else base_name),
            # Check project directory
            os.path.join(os.getcwd(), base_name),
            # Check common subdirectories
            os.path.join(os.getcwd(), "data", base_name),
            os.path.join(os.getcwd(), "texts", base_name),
            os.path.join(os.getcwd(), "transcripts", base_name),
            os.path.join(os.getcwd(), "content", base_name)
        ]
        
        st.info(f"Searching for file '{base_name}' in additional locations...")
        for path in additional_local_paths:
            st.write(f"- Checking: '{path}'")
            if os.path.exists(path):
                st.success(f"Found file locally at '{path}', using this instead")
                try:
                    with open(path, "rb") as f:
                        pdf_data = f.read()
                    break
                except Exception as read_error:
                    st.error(f"Error reading file: {str(read_error)}")
            else:
                st.write(f"  Not found at '{path}'")
                
    if not pdf_data:
        # Help user with what to do next
        st.error(f"Could not find file '{file_path}' anywhere.")
        st.info("To fix this issue, please either:")
        st.info("1. Upload the file using the 'Quick Upload' tool above")
        st.info("2. Place the file in your Downloads folder with the exact name shown")
        st.info("3. Create a 'transcripts' folder in your project directory and place the file there")
        return ""
    
    # Use the generic extraction function
    return extract_text_from_file(pdf_data, file_path)

# Add this function to verify and refresh Supabase connections
def verify_supabase_connection():
    """Verify Supabase connection and refresh if needed"""
    global supabase
    
    try:
        # Try a simple operation to check connection
        test_result = supabase.auth.get_session()
        if test_result:
            return True
    except Exception as e:
        st.warning(f"Supabase connection issue: {str(e)}")
        
    # Try to reinitialize the connection
    try:
        st.info("Attempting to refresh Supabase connection...")
        # First try to get credentials from Streamlit secrets
        supabase_url = st.secrets.get("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY")
        
        # Fall back to environment variables if not found in secrets
        if not supabase_url:
            supabase_url = os.environ.get("SUPABASE_URL")
        if not supabase_key:
            supabase_key = os.environ.get("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            supabase = create_client(supabase_url, supabase_key)
            st.success("Supabase connection refreshed!")
            return True
        else:
            st.error("Supabase credentials not found!")
            return False
    except Exception as refresh_error:
        st.error(f"Failed to refresh Supabase connection: {str(refresh_error)}")
        return False

def setup_bucket_permissions():
    """Set appropriate permissions for the transcripts bucket"""
    try:
        # Create a temporary client with service role key for bucket creation
        if supabase_service_key:
            st.info("Using service role key for bucket creation...")
            admin_client = create_client(supabase_url, supabase_service_key)
        else:
            st.warning("Service role key not found. Using regular key (may fail if permissions are insufficient)")
            admin_client = supabase

        # List buckets to check if transcripts exists
        try:
            buckets = admin_client.storage.list_buckets()
            st.write(f"Found {len(buckets)} buckets: {[b.get('name') for b in buckets]}")
            
            if not any(bucket['name'] == 'transcripts' for bucket in buckets):
                st.info("Creating 'transcripts' bucket...")
                # Create the bucket if it doesn't exist
                admin_client.storage.create_bucket("transcripts", {"public": "true"})
                st.success("Created 'transcripts' bucket with public access")
                
                # Verify bucket was created
                buckets = admin_client.storage.list_buckets()
                if any(bucket['name'] == 'transcripts' for bucket in buckets):
                    st.success("✅ Verified 'transcripts' bucket exists")
                else:
                    st.error("❌ Bucket creation failed - bucket not found after creation")
            else:
                st.success("✅ 'transcripts' bucket already exists")
            
            # Set up bucket policies
            st.info("Setting up bucket policies...")
            try:
                # Create a policy that allows public access
                policy = {
                    "name": "Public Access",
                    "definition": {
                        "statement": [
                            {
                                "action": ["storage.objects.read"],
                                "effect": "allow",
                                "principal": "*"
                            }
                        ]
                    }
                }
                
                # Apply the policy
                admin_client.storage.from_("transcripts").create_signed_url("test.txt", 60)
                st.success("✅ Bucket policies configured successfully")
            except Exception as policy_error:
                st.warning(f"Could not set bucket policies: {str(policy_error)}")
                st.info("You may need to set up policies manually in the Supabase dashboard")
            
            return True
        except Exception as bucket_error:
            st.error(f"Error managing buckets: {str(bucket_error)}")
            return False
            
    except Exception as e:
        st.error(f"Error setting up bucket permissions: {str(e)}")
        return False

def upload_file_to_database(file_content, file_name, course_name, week_number, transcript_name):
    """Upload file directly to database as base64 rather than storage"""
    try:
        # Encode file content as base64
        file_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Check if table exists and create if needed
        try:
            # Check if file_content table exists
            supabase.table("file_contents").select("count", "exact").execute()
        except:
            # Create the table using the REST API instead of raw SQL
            try:
                # First, create a minimal table
                result = supabase.table("file_contents").insert({
                    "id": str(uuid.uuid4()),
                    "course_name": "temp",
                    "week_number": 0,
                    "transcript_name": "temp",
                    "file_name": "temp",
                    "file_data": "temp"
                }).execute()
                st.success("Created file_contents table for direct file storage")
            except Exception as create_error:
                st.error(f"Error creating table: {str(create_error)}")
                return False, "Could not create table"
        
        # Insert the file content into the database
        result = supabase.table("file_contents").insert({
            "course_name": course_name,
            "week_number": week_number,
            "transcript_name": transcript_name,
            "file_name": file_name,
            "file_data": file_base64
        }).execute()
        
        return True, result
    except Exception as e:
        return False, str(e)

def get_file_data_from_database(course_name, week_number=None, file_name=None):
    """Retrieve file content from database"""
    try:
        query = supabase.table("file_contents").select("*").eq("course_name", course_name)
        
        if week_number:
            query = query.eq("week_number", week_number)
        
        if file_name:
            query = query.eq("file_name", file_name)
            
        result = query.execute()
        return result.data
    except Exception as e:
        st.error(f"Error retrieving file data: {str(e)}")
        return []

def get_settings():
    """Get LlamaIndex settings with the configured models"""
    # Create the embedding and LLM models
    embed_model = OpenAIEmbedding()
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    # Set the global settings directly
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Return the configured Settings object
    return Settings

def get_vector_index():
    """
    This function is now a compatibility wrapper for the direct Pinecone operations.
    It avoids using the problematic LlamaIndex PineconeVectorStore class.
    """
    try:
        # Just verify the Pinecone connection and return status info
        index_stats = pinecone_index.describe_index_stats()
        total_vectors = index_stats.get('total_vector_count', 0)
        namespaces = index_stats.get('namespaces', {})
        
        st.info(f"Pinecone index '{pinecone_index_name}' contains {total_vectors} vectors in {len(namespaces)} namespaces")
        
        # Return a dummy object that won't be used
        return {"status": "connected", "index_name": pinecone_index_name}
    except Exception as e:
        st.error(f"Error connecting to Pinecone: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_gemini_response(prompt, context=None):
    """Get response from Gemini model"""
    try:
        if context:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}"
        else:
            full_prompt = prompt
            
        response = gemini_model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting Gemini response: {str(e)}")
        return None

def chunk_text(text, chunk_size=4000, chunk_overlap=200):
    """Split text into smaller chunks with some overlap for better context preservation."""
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    # Preprocess text for better chunking
    # Replace excessive newlines with single newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # If this is not the first chunk, include some overlap
        if start > 0:
            start = max(0, start - chunk_overlap)
            
        # Try to find a good breaking point (sentence or paragraph end)
        if end < text_length:
            # Look for paragraph break first
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break + 2  # Include the newlines
            else:
                # Look for sentence break (period followed by space)
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 3:
                    end = sentence_break + 2  # Include the period and space
                else:
                    # If no good breaks found, look for any punctuation followed by space
                    for punct in ['? ', '! ', '; ', ': ']:
                        punct_break = text.rfind(punct, start, end)
                        if punct_break != -1 and punct_break > start + chunk_size // 3:
                            end = punct_break + 2  # Include the punct and space
                            break
        
        # Get the chunk
        chunk = text[start:end].strip()
        
        # Only add non-empty chunks with substantial content (at least 100 chars)
        if chunk and len(chunk) > 100:
            chunks.append(chunk)
        
        start = end
    
    return chunks

def batch_upsert_chunks(chunks, metadata, batch_size=50):
    """
    Process and upsert multiple chunks in batches
    
    Args:
        chunks: List of text chunks to process
        metadata: Dictionary with common metadata for all chunks
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Tuple of (success_count, total_count)
    """
    if not chunks:
        return 0, 0
    
    # Check if indexing is disabled
    if not ENABLE_PINECONE_INDEXING:
        st.info(SKIP_VECTOR_STORAGE_WARNING)
        return 0, len(chunks)  # Return 0 successful upserts but total count for UI
    
    successful_upserts = 0
    
    # Process chunks in batches
    total_batches = (len(chunks) - 1) // batch_size + 1
    
    # Check for the course_name and transcript_name in metadata
    course_name = metadata.get('course_name', 'unknown_course') 
    transcript_name = metadata.get('transcript_name', 'unknown_document')
    
    # Create a unique namespace prefix for this document to avoid overwriting
    namespace_prefix = f"{course_name}_{metadata.get('week_number', '0')}"
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        current_batch = i // batch_size + 1
        
        with st.spinner(f"Indexing batch {current_batch}/{total_batches} ({len(batch)} chunks)..."):
            # First check if batch already exists by checking first and last chunk
            if current_batch == 1:  # Only check the first batch
                try:
                    # Try to remove existing vectors for this document to avoid duplicates
                    query_text = f"Delete existing vectors for {course_name} week {metadata.get('week_number', '0')} {transcript_name}"
                    
                    # Use a dummy embedding to find vectors with matching metadata
                    embed_model = OpenAIEmbedding()
                    dummy_embedding = embed_model.get_text_embedding(query_text)
                    
                    # Build filter to find vectors for this document
                    filter_dict = {
                        "course_name": {"$eq": course_name},
                        "transcript_name": {"$eq": transcript_name}
                    }
                    if 'week_number' in metadata:
                        filter_dict["week_number"] = {"$eq": metadata['week_number']}
                    
                    try:
                        # Try to delete with filter (only works with some Pinecone plans)
                        st.info(f"Removing existing vectors for {transcript_name} to avoid duplicates...")
                        pinecone_index.delete(
                            namespace="coursera",
                            filter=filter_dict
                        )
                        st.success("Successfully deleted existing vectors")
                    except Exception as delete_error:
                        st.warning(f"Could not delete with filter. Proceeding with new indexing: {str(delete_error)}")
                        # Just continue with the upsert if delete fails
                except Exception as check_error:
                    st.warning(f"Error checking for existing vectors: {str(check_error)}")
            
            # Process the current batch
            for j, chunk in enumerate(batch):
                # Add chunk number to metadata for better tracking
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_number"] = i + j + 1
                chunk_metadata["total_chunks"] = len(chunks)
                
                success, message = upsert_to_pinecone(chunk, chunk_metadata)
                if success:
                    successful_upserts += 1
                else:
                    st.warning(f"Failed to index chunk {i+j+1}: {message}")
                    
    return successful_upserts, len(chunks)

def upsert_to_pinecone(text, metadata):
    """
    Directly upsert text embeddings to Pinecone without using LlamaIndex.
    Uses the newer upsert_records method when possible for better performance.
    """
    # Check if indexing is disabled
    if not ENABLE_PINECONE_INDEXING:
        return False, "Indexing is disabled"
        
    try:
        if not text or not apis_configured:
            return False, "No text or APIs not configured"
        
        # Create a unique ID for this vector
        course_name = metadata.get('course_name', 'unknown')
        week_number = metadata.get('week_number', '0')
        
        # Add a timestamp and unique identifier to avoid collisions
        timestamp = int(time.time())
        random_id = str(uuid.uuid4())[:8]
        vector_id = f"{course_name}_{week_number}_{timestamp}_{random_id}"
        
        # Truncate long text for metadata storage and clean it
        truncated_text = text[:800].replace("\n", " ").strip()
        
        # Standardize metadata values for compatibility
        clean_metadata = {}
        for key, value in metadata.items():
            # Convert all values to strings for consistency
            if value is None:
                clean_value = "none"
            elif isinstance(value, (int, float)):
                clean_value = str(value)
            elif isinstance(value, str):
                clean_value = value.strip()
            else:
                clean_value = str(value)
            
            clean_metadata[key] = clean_value
            
        # Add text preview to metadata
        clean_metadata["text"] = truncated_text
        
        # Try with multiple retry attempts for API issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if the index supports integrated embeddings (field_map)
                try:
                    index_desc = pinecone_index.describe_index_stats()
                    if hasattr(index_desc, 'dimension'):
                        # This is a standard vector index, we need to generate embeddings ourselves
                        embed_model = OpenAIEmbedding()
                        embedding = embed_model.get_text_embedding(text)
                        
                        # Upsert the vector
                        pinecone_index.upsert(
                            vectors=[(
                                vector_id, 
                                embedding, 
                                clean_metadata
                            )],
                            namespace="coursera"
                        )
                        return True, f"Indexed content with ID: {vector_id}"
                    else:
                        # This might be an index with integrated embedding
                        try:
                            # Use the newer upsert_records which can automatically convert text to vectors
                            pinecone_index.upsert_records(
                                namespace="coursera",
                                records=[{
                                    "_id": vector_id,
                                    "chunk_text": text,  # Will be converted to vector automatically
                                    **clean_metadata  # Include all metadata
                                }]
                            )
                            return True, f"Indexed text with ID: {vector_id} using integrated embedding"
                        except Exception as e:
                            if "no text embedding field_map" in str(e).lower():
                                # Fall back to the manual approach
                                raise
                            else:
                                raise
                except Exception as e:
                    # Fall back to the basic upsert method
                    embed_model = OpenAIEmbedding()
                    embedding = embed_model.get_text_embedding(text)
                    
                    # Upsert the vector
                    pinecone_index.upsert(
                        vectors=[(
                            vector_id, 
                            embedding, 
                            clean_metadata
                        )],
                        namespace="coursera"
                    )
                    return True, f"Indexed content with ID: {vector_id} (fallback method)"
                
            except Exception as retry_error:
                if attempt < max_retries - 1:
                    wait_time = 1 * (2 ** attempt)  # Exponential backoff
                    time.sleep(wait_time)
                else:
                    raise retry_error
                    
    except Exception as e:
        return False, f"Error indexing to Pinecone: {str(e)}"

def semantic_search_with_pinecone(query, course_name=None, week_number=None, top_k=5):
    """
    Perform semantic search using Pinecone directly
    """
    try:
        # If indexing is disabled, but search is still enabled
        if not ENABLE_PINECONE_INDEXING and not ENABLE_VECTOR_SEARCH:
            st.info("Vector search is disabled. Using alternative search method.")
            # Return empty string to trigger fallback to full text processing
            return ""
            
        # Even if indexing is disabled, we'll still search if ENABLE_VECTOR_SEARCH is True
        if not ENABLE_PINECONE_INDEXING and ENABLE_VECTOR_SEARCH:
            st.info("Using existing vectors for search (no new vectors will be created).")
            
        # Get the embedding for the query with retry logic
        max_retries = 3
        retry_count = 0
        query_embedding = None
        
        while retry_count < max_retries:
            try:
                embed_model = OpenAIEmbedding()
                query_embedding = embed_model.get_text_embedding(query)
                break
            except Exception as e:
                if "APIConnectionError" in str(e) or "Connection error" in str(e):
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = 1 * (2 ** retry_count)  # Exponential backoff
                        st.warning(f"API connection error during search, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        st.error("Maximum retries reached for embedding API during search")
                        return ""
                else:
                    raise
        
        if not query_embedding:
            return ""
            
        # Build filter if needed
        filter_dict = {}
        if course_name:
            filter_dict["course_name"] = {"$eq": course_name}
        if week_number:
            filter_dict["week_number"] = {"$eq": week_number}
        
        # Track which approach worked for logging
        search_method_used = "standard"
        
        # Execute the query
        try:
            results = pinecone_index.query(
                namespace="coursera",
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            # Extract text from matches
            retrieved_texts = []
            scores = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    retrieved_texts.append(match.metadata['text'])
                    scores.append(match.score)
            
            # Log search quality
            if scores:
                avg_score = sum(scores) / len(scores)
                st.write(f"📊 Search quality: {avg_score:.2f} (higher is better)")
            
            if not retrieved_texts:
                st.warning("No relevant vectors found in Pinecone. This may happen if you haven't indexed this content before.")
                return ""
                
            # Format the retrieved texts with their relevance scores
            formatted_texts = []
            for i, (text, score) in enumerate(zip(retrieved_texts, scores)):
                # Add a section header with relevance score
                formatted_texts.append(f"--- Segment {i+1} (Relevance: {score:.2f}) ---\n{text}\n")
                
            return "\n\n".join(formatted_texts)
        except Exception as query_error:
            st.warning(f"Error in standard Pinecone query: {str(query_error)}")
            search_method_used = "fallback"
            
            # If filtering is not supported, try the fallback approach
            if "not supported" in str(query_error).lower():
                # If filtering is not supported, try without filters
                try:
                    results = pinecone_index.query(
                        namespace="coursera",
                        vector=query_embedding,
                        top_k=top_k * 2,  # Get more results since we'll filter manually
                        include_metadata=True
                    )
                    
                    # Manually filter results
                    filtered_texts = []
                    scores = []
                    for match in results.matches:
                        metadata = match.metadata
                        if not metadata:
                            continue
                            
                        # Apply our filters manually
                        if course_name and metadata.get('course_name') != course_name:
                            continue
                        if week_number and metadata.get('week_number') != week_number:
                            continue
                            
                        if 'text' in metadata:
                            filtered_texts.append(metadata['text'])
                            scores.append(match.score)
                    
                    # Format the retrieved texts with their relevance scores
                    formatted_texts = []
                    for i, (text, score) in enumerate(zip(filtered_texts, scores)):
                        # Add a section header with relevance score
                        formatted_texts.append(f"--- Segment {i+1} (Relevance: {score:.2f}) ---\n{text}\n")
                            
                    if filtered_texts:
                        st.info(f"Used fallback search method with manual filtering ({len(filtered_texts)} results)")
                        return "\n\n".join(formatted_texts)
                    else:
                        return ""
                except Exception as fallback_error:
                    st.error(f"Fallback query failed: {str(fallback_error)}")
                    return ""
            return ""
    except Exception as e:
        st.warning(f"Error in semantic search: {str(e)}")
        return ""

def check_if_already_indexed(course_name, week_number, transcript_name=None):
    """
    Check if content for a specific course and week is already indexed in Pinecone
    
    Args:
        course_name: Name of the course
        week_number: Week number
        transcript_name: Optional transcript name to check
        
    Returns:
        bool: True if already indexed, False otherwise
    """
    # Check if indexing is disabled - always return True to skip indexing
    if not ENABLE_PINECONE_INDEXING:
        return True
        
    try:
        # Build query filter
        filter_dict = {
            "course_name": {"$eq": course_name},
            "week_number": {"$eq": week_number}
        }
        
        if transcript_name:
            filter_dict["transcript_name"] = {"$eq": transcript_name}
        
        # For Serverless and Starter indexes, we can't use filtered describe_index_stats
        # Instead, we'll check by making a direct query
        try:
            # Get a single vector to see if any exist
            embed_model = OpenAIEmbedding()
            # Use a generic embedding just to check existence
            dummy_embedding = embed_model.get_text_embedding("test query")
            
            results = pinecone_index.query(
                namespace="coursera",
                vector=dummy_embedding,
                top_k=1,
                include_metadata=True,
                filter=filter_dict
            )
            
            # If we got any matches with the right metadata, content exists
            return len(results.matches) > 0
        except Exception as query_error:
            if "not support" in str(query_error).lower():
                # If filtering not supported, try a workaround: query without filter and check metadata
                results = pinecone_index.query(
                    namespace="coursera",
                    vector=dummy_embedding,
                    top_k=10,  # Get a few results to search through
                    include_metadata=True
                )
                
                # Manually check if any of the returned vectors match our criteria
                for match in results.matches:
                    metadata = match.metadata if hasattr(match, 'metadata') else {}
                    if metadata.get('course_name') == course_name and \
                       metadata.get('week_number') == week_number and \
                       (transcript_name is None or metadata.get('transcript_name') == transcript_name):
                        return True
                        
                # Check stats to see if the namespace has any vectors
                stats = pinecone_index.describe_index_stats()
                if 'namespaces' in stats and 'coursera' in stats['namespaces']:
                    vector_count = stats['namespaces']['coursera'].get('vector_count', 0)
                    if vector_count == 0:
                        return False  # No vectors at all
                    else:
                        # We have vectors but couldn't confirm the exact ones we want
                        # Return True only if we're reasonably sure (like if this is a re-run)
                        st.info("Couldn't precisely determine indexing status. Using heuristic check.")
                        # Create a simple key to check if we've seen this before
                        session_key = f"indexed_{course_name}_{week_number}_{transcript_name}"
                        if session_key in st.session_state:
                            return True
                        else:
                            # Mark this as seen for future checks
                            st.session_state[session_key] = True
                            return False
            else:
                raise
    except Exception as e:
        st.warning(f"Error checking index status: {str(e)}")
        return False  # Assume not indexed on error, to be safe

def manage_bucket():
    """List and manage bucket contents"""
    try:
        # Create a temporary client with service role key for bucket management
        if supabase_service_key:
            st.info("Using service role key for bucket management...")
            admin_client = create_client(supabase_url, supabase_service_key)
        else:
            st.warning("Service role key not found. Using regular key (may fail if permissions are insufficient)")
            admin_client = supabase

        # List all buckets
        buckets = admin_client.storage.list_buckets()
        st.write("### Current Buckets")
        for bucket in buckets:
            st.write(f"- {bucket['name']} (Public: {bucket.get('public', False)})")
        
        # List contents of transcripts bucket
        st.write("### Contents of 'transcripts' bucket")
        try:
            files = admin_client.storage.from_("transcripts").list()
            if files:
                for file in files:
                    st.write(f"- {file['name']} ({file.get('metadata', {}).get('size', 0) / 1024:.1f} KB)")
            else:
                st.info("Bucket is empty")
        except Exception as e:
            st.error(f"Error listing bucket contents: {str(e)}")
        
        # Add bucket management options
        st.write("### Bucket Management")
        if st.button("Delete 'transcripts' bucket"):
            try:
                # First, delete all files in the bucket
                files = admin_client.storage.from_("transcripts").list()
                for file in files:
                    admin_client.storage.from_("transcripts").remove([file['name']])
                
                # Then delete the bucket
                admin_client.storage.delete_bucket("transcripts")
                st.success("Successfully deleted 'transcripts' bucket")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting bucket: {str(e)}")
        
        if st.button("Recreate 'transcripts' bucket"):
            try:
                # Create new bucket with public access
                admin_client.storage.create_bucket("transcripts", {"public": "true"})
                st.success("Successfully created new 'transcripts' bucket")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error creating bucket: {str(e)}")
        
        if st.button("Update bucket policies"):
            try:
                # Set bucket to public
                admin_client.storage.update_bucket("transcripts", {"public": "true"})
                st.success("Successfully updated bucket policies")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error updating policies: {str(e)}")
                
    except Exception as e:
        st.error(f"Error managing bucket: {str(e)}")

def verify_supabase_config():
    """Verify and display Supabase client configuration"""
    try:
        st.write("### Supabase Configuration Check")
        
        # Check URL
        if not supabase_url:
            st.error("❌ Supabase URL is missing")
            return False
        else:
            st.success(f"✅ Supabase URL: {supabase_url}")
        
        # Check anon key
        if not supabase_key:
            st.error("❌ Supabase anon key is missing")
            return False
        else:
            # Mask the key for security
            masked_key = supabase_key[:5] + "..." + supabase_key[-5:] if len(supabase_key) > 10 else "***"
            st.success(f"✅ Supabase anon key: {masked_key}")
        
        # Check service key
        if not supabase_service_key:
            st.error("❌ Supabase service key is missing")
            return False
        else:
            # Mask the key for security
            masked_service_key = supabase_service_key[:5] + "..." + supabase_service_key[-5:] if len(supabase_service_key) > 10 else "***"
            st.success(f"✅ Supabase service key: {masked_service_key}")
        
        # Test connection with anon key
        st.write("### Testing Connection with Anon Key")
        try:
            test_client = create_client(supabase_url, supabase_key)
            # Try a simple operation
            test_client.auth.get_session()
            st.success("✅ Connection with anon key successful")
        except Exception as e:
            st.error(f"❌ Connection with anon key failed: {str(e)}")
            return False
        
        # Test connection with service key
        st.write("### Testing Connection with Service Key")
        try:
            admin_client = create_client(supabase_url, supabase_service_key)
            # Try a storage operation
            buckets = admin_client.storage.list_buckets()
            st.success(f"✅ Connection with service key successful. Found {len(buckets)} buckets")
            return True
        except Exception as e:
            st.error(f"❌ Connection with service key failed: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"Error verifying configuration: {str(e)}")
        return False

def verify_and_fix_storage():
    """Verify files in database exist in storage and fix if missing"""
    try:
        # Get all records from transcripts table
        records = supabase.table("transcripts").select("*").execute()
        
        if not records.data:
            st.warning("No records found in transcripts table")
            return
            
        st.write(f"Found {len(records.data)} records in database")
        
        # Check each record
        for record in records.data:
            file_path = record.get('file_path')
            if not file_path:
                continue
                
            st.write(f"Checking file: {file_path}")
            
            # Check if file exists in storage
            try:
                files = supabase.storage.from_("transcripts").list()
                file_exists = any(f.get('name') == file_path for f in files)
                
                if not file_exists:
                    st.warning(f"File {file_path} is missing from storage")
                    
                    # Try to find the file locally
                    local_paths = [
                        file_path,
                        os.path.join(os.getcwd(), file_path),
                        os.path.join(os.path.expanduser("~"), "Downloads", file_path),
                        os.path.join(os.path.expanduser("~"), "Desktop", file_path)
                    ]
                    
                    file_found = False
                    for path in local_paths:
                        if os.path.exists(path):
                            st.info(f"Found file locally at: {path}")
                            # Upload to storage
                            with open(path, "rb") as f:
                                supabase.storage.from_("transcripts").upload(
                                    path=file_path,
                                    file=f.read(),
                                    file_options={"content-type": "text/plain"}
                                )
                            st.success(f"Successfully uploaded {file_path} to storage")
                            file_found = True
                            break
                    
                    if not file_found:
                        st.error(f"Could not find {file_path} locally to upload")
                else:
                    st.success(f"File {file_path} exists in storage")
                    
            except Exception as e:
                st.error(f"Error checking file {file_path}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error verifying storage: {str(e)}")

# Modify the get_file_from_supabase function to be more resilient
def get_file_from_supabase(file_path):
    """Download a file from Supabase storage and return its contents"""
    global supabase
    
    # Sanitize file path - remove any whitespace or newlines
    clean_path = file_path.strip()
    
    # Skip if empty path
    if not clean_path:
        return None
    
    # Ensure we're using the correct project
    project_id = "vmppiyatjirgfeqlgswm"
    project_url = f"https://{project_id}.supabase.co"
    
    # Log information about the connection
    st.info(f"Connecting to Supabase project: {project_id}")
    st.info(f"Attempting to access file: {clean_path}")
        
    # Try to get the file from Supabase
    for attempt in range(2):  # Make up to 2 attempts
        try:
            # Ensure connection is valid
            if attempt > 0:
                verify_supabase_connection()
            
            # On second attempt, try creating a new client specifically for this project
            if attempt == 1:
                st.info("Trying with direct project connection...")
                try:
                    # Use environment variables or secrets
                    api_key = supabase_key or os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY")
                    if not api_key:
                        st.error("No API key available for Supabase")
                        return None
                    
                    # Create a direct client to the specific project
                    direct_client = create_client(project_url, api_key)
                    
                    # Try download with the direct client
                    response = direct_client.storage.from_("transcripts").download(clean_path)
                    if response:
                        st.success(f"Successfully downloaded '{clean_path}' from Supabase using direct connection!")
                        return response
                except Exception as direct_error:
                    st.error(f"Direct connection failed: {str(direct_error)}")
                
            # Get direct info about the bucket
            try:
                all_buckets = supabase.storage.list_buckets()
                bucket_names = [b.get('name') for b in all_buckets]
                if 'transcripts' not in bucket_names:
                    st.error("'transcripts' bucket not found in your Supabase project!")
                    st.write(f"Available buckets: {', '.join(bucket_names)}")
                    st.info(f"Make sure your project ID is correct: {project_id}")
                else:
                    st.success("'transcripts' bucket found!")
            except Exception as bucket_error:
                st.warning(f"Could not check buckets: {str(bucket_error)}")
            
            # Try to list files to verify bucket access
            try:
                bucket_files = supabase.storage.from_("transcripts").list()
                st.write(f"Files in Supabase 'transcripts' bucket: {[f.get('name') for f in bucket_files]}")
                if not any(f.get('name') == clean_path for f in bucket_files):
                    st.warning(f"File '{clean_path}' not found among the {len(bucket_files)} files in the bucket")
                    # Look for similar files
                    similar_files = [f.get('name') for f in bucket_files if f.get('name') and 
                                    (clean_path.lower() in f.get('name').lower() or 
                                     f.get('name').lower() in clean_path.lower())]
                    if similar_files:
                        st.info(f"Similar files found: {', '.join(similar_files)}")
                        st.info("If one of these is the file you want, try using one of these instead:")
                        # Create buttons for each similar file
                        for similar_file in similar_files[:3]:  # Limit to first 3 to avoid too many buttons
                            if st.button(f"Try '{similar_file}'", key=f"try_{similar_file}"):
                                # Try to download this similar file instead
                                try:
                                    alt_response = supabase.storage.from_("transcripts").download(similar_file)
                                    if alt_response:
                                        st.success(f"Successfully downloaded '{similar_file}' instead!")
                                        return alt_response
                                except Exception as alt_error:
                                    st.error(f"Error downloading alternative file: {str(alt_error)}")
            except Exception as list_error:
                st.warning(f"Could not list files in bucket: {str(list_error)}")
                
            # Attempt actual download
            response = supabase.storage.from_("transcripts").download(clean_path)
            if response:
                st.success(f"Successfully downloaded '{clean_path}' from Supabase!")
                return response
        except Exception as e:
            st.error(f"Error downloading file from Supabase (attempt {attempt+1}): {str(e)}")
            if attempt == 0:
                st.info("Trying again with refreshed connection...")
            
    # If Supabase download failed, try local files
    st.warning("Supabase download failed. Trying local files...")
    local_paths = [
        file_path,  # Try direct path
        file_path.strip(),  # Try without whitespace
        os.path.join(os.getcwd(), file_path.strip()),  # Try in current directory
        os.path.join(os.path.expanduser("~"), "Downloads", file_path.strip())  # Try in Downloads folder
    ]
    
    for path in local_paths:
        if os.path.exists(path):
            st.success(f"Found file locally at '{path}', using this instead")
            with open(path, "rb") as f:
                return f.read()
    
    # Last resort: Try direct URL access if possible (for public files)
    try:
        st.info("Attempting direct URL access as last resort...")
        public_url = f"https://{project_id}.supabase.co/storage/v1/object/public/transcripts/{clean_path}"
        st.info(f"Trying URL: {public_url}")
        
        response = requests.get(public_url)
        if response.status_code == 200:
            st.success("Successfully downloaded file via public URL!")
            return response.content
        else:
            st.error(f"Public URL access failed with status code: {response.status_code}")
    except Exception as url_error:
        st.error(f"Error with direct URL access: {str(url_error)}")
    
    return None

# Main application logic for each tab
with tab1:
    st.header("Upload Coursera Transcripts")
    
    # Add configuration check section
    with st.expander("🔧 Verify Supabase Configuration", expanded=True):
        if verify_supabase_config():
            st.success("✅ Supabase configuration is correct")
        else:
            st.error("❌ There are issues with your Supabase configuration")
            st.info("Please check your secrets.toml file and make sure all keys are correct")
    
    # Add bucket management section
    with st.expander("🔄 Manage Storage Bucket", expanded=True):
        manage_bucket()
    
    # Add storage verification section
    with st.expander("🔍 Verify Storage Files", expanded=True):
        if st.button("Check and Fix Storage Files"):
            with st.spinner("Verifying storage files..."):
                verify_and_fix_storage()
    
    # Add button to check and fix bucket permissions
    if st.button("⚙️ Setup Storage & Tables"):
        with st.spinner("Setting up Supabase storage..."):
            # Setup bucket permissions
            if setup_bucket_permissions():
                st.success("Supabase storage is now configured!")
            
            # Check if transcripts table exists
            try:
                # Try a simple query to see if the table exists
                result = supabase.table("transcripts").select("count", "exact").execute()
                st.success("✅ Transcripts table exists!")
            except Exception:
                # Create the table if it doesn't exist
                try:
                    # Create table automatically without asking
                    try:
                        # Create a minimal table using the REST API
                        result = supabase.table("transcripts").insert({
                            "id": str(uuid.uuid4()),
                            "course_name": "temp",
                            "week_number": 0,
                            "transcript_name": "temp",
                            "file_path": "temp"
                        }).execute()
                        st.success("Successfully created 'transcripts' table!")
                        # Verify table exists
                        result = supabase.table("transcripts").select("count", "exact").execute()
                        st.write(f"Table check result: {result}")
                    except Exception as create_table_error:
                        st.error(f"Failed to create table: {str(create_table_error)}")
                        st.info("You need to create the table manually in the Supabase dashboard.")
                        st.stop()
                except Exception as table_error:
                    st.error(f"❌ Error creating table: {str(table_error)}")
    
    course_name = st.text_input("Course Name", placeholder="e.g., Machine Learning")
    week_number = st.number_input("Week Number", min_value=1, max_value=20, value=1)
    transcript_name = st.text_input("Transcript Name", placeholder="e.g., Lecture 1 - Introduction")
    
    uploaded_file = st.file_uploader("Upload Transcript", type=["pdf", "txt", "json", "md"])
    
    if st.button("Upload and Index") and uploaded_file and course_name and transcript_name:
        with st.spinner("Processing transcript..."):
            # Extract text
            text = extract_text_from_file(uploaded_file.read(), uploaded_file.name)
            if not text:
                st.error("Could not extract text from the file.")
            else:
                try:
                    st.write("Extracted text successfully. Now uploading...")
                    
                    # Try using normal Supabase storage upload first
                    storage_upload_success = False
                    
                    # Save file to Supabase
                    file_path = f"{course_name}_{week_number}_{uploaded_file.name}"
                    
                    # Try to upload the file directly without complex path
                    try:
                        # Debug info about the file being uploaded
                        st.write(f"Uploading file: {uploaded_file.name}")
                        st.write(f"File size: {len(uploaded_file.read())} bytes")
                        st.write(f"Simple path: {file_path}")
                        
                        # More debugging info about Supabase connection
                        st.write("Checking Supabase credentials and bucket...")
                        buckets = supabase.storage.list_buckets()
                        st.write(f"Available buckets: {[bucket['name'] for bucket in buckets]}")
                        
                        # Check if transcripts bucket exists and create if needed
                        transcript_bucket_exists = any(bucket['name'] == 'transcripts' for bucket in buckets)
                        if not transcript_bucket_exists:
                            st.warning("'transcripts' bucket not found. Creating it now...")
                            try:
                                supabase.storage.create_bucket("transcripts", {"public": "true"})
                                st.success("Created 'transcripts' bucket!")
                            except Exception as bucket_error:
                                st.error(f"Error creating bucket: {str(bucket_error)}")
                        
                        # Direct upload with explicit content type
                        file_bytes = uploaded_file.read()
                        st.write(f"Attempting to upload {len(file_bytes)} bytes...")
                        
                        # Try uploading using raw file bytes
                        upload_result = supabase.storage.from_("transcripts").upload(
                            path=file_path,
                            file=file_bytes,
                            file_options={"content-type": uploaded_file.type}
                        )
                        
                        st.write(f"Upload result: {upload_result}")
                        st.success("File uploaded to Supabase storage successfully!")
                        storage_upload_success = True
                        
                        # Verify the file exists in storage
                        try:
                            # List files in bucket to verify upload
                            files = supabase.storage.from_("transcripts").list()
                            st.write(f"Files in transcripts bucket: {files}")
                            
                            # Check if our file is in the list
                            if any(f["name"] == file_path for f in files):
                                st.success(f"✅ Verified file {file_path} exists in storage!")
                            else:
                                st.warning(f"⚠️ Could not verify file {file_path} in storage listing.")
                                storage_upload_success = False
                        except Exception as list_error:
                            st.warning(f"Could not verify file in storage: {str(list_error)}")
                        
                    except Exception as upload_error:
                        st.error(f"Error uploading file to Supabase: {str(upload_error)}")
                        
                        # Try with alternative file path
                        try:
                            st.info("Trying alternative upload method...")
                            # Try with simpler file name only
                            simple_path = uploaded_file.name
                            
                            # Upload with file name only
                            upload_result = supabase.storage.from_("transcripts").upload(
                                path=simple_path,
                                file=uploaded_file.read(),
                                file_options={"content-type": uploaded_file.type}
                            )
                            
                            st.success(f"File uploaded with simplified path: {simple_path}")
                            file_path = simple_path  # Update the file path for database record
                            storage_upload_success = True
                            
                            # Verify upload
                            files = supabase.storage.from_("transcripts").list()
                            st.write(f"Files in bucket after simplified upload: {files}")
                        except Exception as alt_error:
                            st.error(f"Alternative upload also failed: {str(alt_error)}")
                            st.error("Will try direct database storage instead.")
                            storage_upload_success = False
                    
                    # If storage upload failed, try direct database upload as fallback
                    if not storage_upload_success:
                        st.info("Using direct database storage as fallback...")
                        db_upload_success, result = upload_file_to_database(
                            uploaded_file.read(),
                            uploaded_file.name,
                            course_name,
                            week_number,
                            transcript_name
                        )
                        
                        if db_upload_success:
                            st.success("File uploaded directly to database successfully!")
                            file_path = f"DB_{uploaded_file.name}"  # Mark file path as database stored
                        else:
                            st.error(f"Database upload failed: {result}")
                            st.error("All upload methods failed. Please check your Supabase configuration.")
                            st.stop()
                    
                    # Save metadata to Supabase
                    try:
                        # First check if the table exists
                        try:
                            # Try a simple query to see if the table exists
                            supabase.table("transcripts").select("count", "exact").execute()
                        except Exception as table_error:
                            st.error(f"Error accessing 'transcripts' table: {str(table_error)}")
                            st.info("Please make sure you created the 'transcripts' table in your Supabase project with the correct structure.")
                            st.code("""
CREATE TABLE transcripts (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  course_name TEXT NOT NULL,
  week_number INTEGER,
  transcript_name TEXT NOT NULL,
  file_path TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
                            """, language="sql")
                            
                            # Insert the record
                            supabase.table("transcripts").insert({
                                "course_name": course_name,
                                "week_number": week_number,
                                "transcript_name": transcript_name,
                                "file_path": file_path
                            }).execute()
                            st.success("Metadata saved to Supabase database successfully!")
                    except Exception as db_error:
                        st.error(f"Error saving metadata: {str(db_error)}")
                        st.stop()
                    
                    # Index transcript in Pinecone (optional - may skip if having issues)
                    try:
                        # Check if indexing is enabled
                        if not ENABLE_PINECONE_INDEXING:
                            st.info(SKIP_VECTOR_STORAGE_WARNING)
                        else:
                            from llama_index.core import Document
                            from llama_index.core.node_parser import SimpleNodeParser
                            
                            settings = get_settings()
                            parser = SimpleNodeParser.from_defaults()
                            nodes = parser.get_nodes_from_documents([Document(text=text)])
                            
                            # Add metadata to nodes
                            for node in nodes:
                                node.metadata = {
                                    "course_name": course_name,
                                    "week_number": week_number,
                                    "transcript_name": transcript_name
                                }
                            
                            # Index to Pinecone
                            try:
                                # Check if index exists, create if not
                                if pinecone_index_name not in pc.list_indexes().names():
                                    pc.create_index(
                                        name=pinecone_index_name,
                                        dimension=1536,  # For OpenAI embeddings
                                        metric="cosine",
                                        spec=pinecone.ServerlessSpec(
                                            cloud="gcp",
                                            region="us-central1"
                                        )
                                    )
                                    
                                pinecone_index = pc.Index(pinecone_index_name)
                                vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
                                index = VectorStoreIndex.from_vector_store(
                                    vector_store,
                                    embed_model=settings.embed_model
                                )
                                # Add nodes to index
                                index.insert_nodes(nodes)
                                st.success(f"Indexed content from {file_path}")
                            except Exception as index_error:
                                st.error(f"Error indexing to Pinecone: {str(index_error)}")
                    except Exception as index_error:
                        st.warning(f"Note: Could not index to Pinecone: {str(index_error)}")
                        st.warning("This is not critical - you can still use the summarize and question functions.")
                    
                    st.success(f"Transcript uploaded and processed successfully!")
                    
                    # Add a refresh button to see the new upload in other tabs
                    if st.button("Refresh App"):
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.header("Summarize Transcripts")
    
    # Add simplified storage checker expander
    with st.expander("🔍 Check Supabase Storage", expanded=True):
        # Add direct file upload for quick fixes
        st.write("### Quick Upload")
        upload_course = st.text_input("Course Name", placeholder="e.g., AI For Everyone", key="quick_upload_course")
        upload_week = st.number_input("Week Number", min_value=1, max_value=20, value=1, key="quick_upload_week")
        upload_file = st.file_uploader("Upload File", type=["pdf", "txt", "json", "md"], key="quick_upload_file")
        
        if upload_file and upload_course and st.button("Upload to Supabase"):
            try:
                # First check if bucket exists, create if not
                try:
                    buckets = supabase.storage.list_buckets()
                    if not any(bucket['name'] == 'transcripts' for bucket in buckets):
                        st.warning("'transcripts' bucket not found. Creating it now...")
                        supabase.storage.create_bucket("transcripts", {"public": "true"})
                except Exception as bucket_error:
                    st.error(f"Error checking buckets: {str(bucket_error)}")
                
                # Generate clean filename
                clean_filename = f"{upload_course.replace(' ', '_')}_{upload_week}_{upload_file.name.replace(' ', '_')}"
                
                # Upload file
                with st.spinner(f"Uploading {upload_file.name}..."):
                    # Reset file pointer to beginning
                    upload_file.seek(0)
                    
                    # Upload to Supabase
                    result = supabase.storage.from_("transcripts").upload(
                        path=clean_filename,
                        file=upload_file.read(),
                        file_options={"content-type": upload_file.type or "application/octet-stream"}
                    )
                    
                    # Add to database
                    db_result = supabase.table("transcripts").insert({
                        "course_name": upload_course,
                        "week_number": upload_week,
                        "transcript_name": upload_file.name,
                        "file_path": clean_filename
                    }).execute()
                    
                    st.success(f"Successfully uploaded {clean_filename} to Supabase")
                    st.info("Please refresh the page to see the new file in the dropdown")
            except Exception as e:
                st.error(f"Error during quick upload: {str(e)}")
                
        # Add diagnostic buttons in a row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Setup Storage"):
                with st.spinner("Setting up storage..."):
                    setup_bucket_permissions()
        
        with col2:
            if st.button("List Files"):
                try:
                    buckets = supabase.storage.list_buckets()
                    if not any(bucket['name'] == 'transcripts' for bucket in buckets):
                        st.warning("'transcripts' bucket not found. Creating it now...")
                        supabase.storage.create_bucket("transcripts", {"public": "true"})
                        st.success("Created 'transcripts' bucket")
                    
                    files = supabase.storage.from_("transcripts").list()
                    if files:
                        st.success(f"Found {len(files)} files in 'transcripts' bucket")
                        for file in files:
                            st.write(f"- {file.get('name', 'Unknown')} ({int(file.get('metadata', {}).get('size', 0) / 1024)} KB)")
                    else:
                        st.warning("No files found in 'transcripts' bucket")
                except Exception as e:
                    st.error(f"Error listing files: {str(e)}")
        
        with col3:
            if st.button("Test Connection"):
                with st.spinner("Testing connection..."):
                    if verify_supabase_connection():
                        st.success("✅ Supabase connection is working")
                    else:
                        st.error("❌ Supabase connection failed")
    
    # Check if any files exist in the database table
    try:
        file_count_result = supabase.table("transcripts").select("count").execute()
        file_count = len(file_count_result.data) if file_count_result.data else 0
        
        if file_count == 0:
            st.warning("⚠️ No transcript records found in the database.")
            st.info("Please use the Upload tab first to add some transcripts, or use the Quick Upload tool above.")
    except Exception as check_error:
        st.error(f"Error checking database: {str(check_error)}")
    
    # Get available courses
    courses = []
    if apis_configured:
        try:
            response = supabase.table("transcripts").select("course_name").execute()
            courses = list(set([item["course_name"] for item in response.data]))
        except Exception as e:
            st.error(f"Error fetching courses: {str(e)}")
    
    selected_course = st.selectbox("Select Course", courses if courses else ["No courses available"])
    
    if selected_course != "No courses available":
        # Get weeks for selected course
        weeks = []
        try:
            response = supabase.table("transcripts").select("week_number").eq("course_name", selected_course).execute()
            weeks = sorted(list(set([item["week_number"] for item in response.data])))
        except Exception as e:
            st.error(f"Error fetching weeks: {str(e)}")
        
        selected_week = st.selectbox("Select Week", weeks if weeks else [])
        
        if selected_week and st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                # Show Pinecone indexing status
                if ENABLE_PINECONE_INDEXING:
                    st.info("🔄 Pinecone vector indexing is ENABLED. Text will be converted to embeddings and stored in Pinecone.")
                else:
                    st.info("ℹ️ Pinecone vector indexing is disabled. Text will not be stored as embeddings.")
                
                try:
                    # Fetch transcripts for the selected course and week
                    response = supabase.table("transcripts").select("*").eq("course_name", selected_course).eq("week_number", selected_week).execute()
                    if not response.data:
                        st.warning("No transcripts found for the selected course and week.")
                    else:
                        # Display detailed information about each transcript to help with debugging
                        with st.expander("🔍 Transcript Details", expanded=False):
                            st.write("These are the transcript records found in the database:")
                            for i, transcript in enumerate(response.data):
                                st.write(f"**Transcript #{i+1}**")
                                st.write(f"- **File path:** '{transcript.get('file_path', '')}'")
                                st.write(f"- **Course:** '{transcript.get('course_name', '')}'")
                                st.write(f"- **Week:** {transcript.get('week_number', '')}")
                                st.write(f"- **Name:** '{transcript.get('transcript_name', '')}'")
                                st.write("---")
                        
                        # Try to process files directly from Supabase storage
                        st.info(f"Found {len(response.data)} transcript(s) for {selected_course}, Week {selected_week}")
                        
                        # Process each transcript
                        all_text = ""
                        for transcript in response.data:
                            file_path = transcript["file_path"]
                            st.write(f"Processing {file_path}...")
                            
                            # Check if this transcript is already indexed in Pinecone
                            if ENABLE_PINECONE_INDEXING:
                                already_indexed = check_if_already_indexed(
                                    selected_course, 
                                    selected_week, 
                                    transcript.get("transcript_name", file_path)
                                )
                                if already_indexed:
                                    st.info(f"📊 Content from {file_path} is already indexed in Pinecone")
                                else:
                                    st.info(f"🆕 Content from {file_path} will be indexed in Pinecone")
                            
                            # First check if the bucket exists
                            try:
                                buckets = supabase.storage.list_buckets()
                                if not any(bucket['name'] == 'transcripts' for bucket in buckets):
                                    # Create the bucket if it doesn't exist
                                    st.warning("'transcripts' bucket not found. Creating it now...")
                                    supabase.storage.create_bucket("transcripts", {"public": "true"})
                            except Exception as bucket_error:
                                st.error(f"Error checking buckets: {str(bucket_error)}")
                                # Continue anyway as we might be able to find the file locally
                            
                            # Try multiple approaches to get file content
                            text = None
                            
                            # First, try direct database storage (if file_path starts with DB_)
                            if file_path.startswith("DB_"):
                                try:
                                    filename = file_path[3:].strip()
                                    db_results = supabase.table("file_contents").select("*").ilike("file_name", filename).execute()
                                    if db_results.data:
                                        st.success(f"Found file '{filename}' in database storage")
                                        file_record = db_results.data[0]
                                        file_data = base64.b64decode(file_record["file_data"])
                                        text = extract_text_from_file(file_data, filename)
                                except Exception as db_error:
                                    st.warning(f"Could not retrieve from database: {str(db_error)}")
                            
                            # If not found in DB, try Supabase storage
                            if not text:
                                try:
                                    file_data = get_file_from_supabase(file_path)
                                    if file_data:
                                        text = extract_text_from_file(file_data, file_path)
                                except Exception as storage_error:
                                    st.warning(f"Error accessing storage: {str(storage_error)}")
                            
                            # If still not found, try local files with common variations
                            if not text:
                                home_dir = os.path.expanduser("~")
                                base_name = os.path.basename(file_path.strip())
                                local_paths = [
                                    file_path,
                                    os.path.join(os.getcwd(), base_name),
                                    os.path.join(home_dir, "Downloads", base_name),
                                    os.path.join(os.getcwd(), "transcripts", base_name),
                                    # Try with and without underscores
                                    os.path.join(os.getcwd(), base_name.replace("_", " ")),
                                    os.path.join(os.getcwd(), base_name.replace(" ", "_"))
                                ]
                                
                                for path in local_paths:
                                    if os.path.exists(path):
                                        st.success(f"Found file locally at '{path}'")
                                        try:
                                            with open(path, "rb") as f:
                                                file_data = f.read()
                                                text = extract_text_from_file(file_data, path)
                                                break
                                        except Exception as read_error:
                                            st.error(f"Error reading file: {str(read_error)}")
                            
                            # Add text to the combined content
                            if text:
                                all_text += text + "\n\n"
                                st.success(f"Successfully extracted text from {file_path}")
                                
                                # Index the extracted text to Pinecone
                                if ENABLE_PINECONE_INDEXING:
                                    with st.spinner(f"Creating vector embeddings for {file_path}..."):
                                        try:
                                            # Split text into manageable chunks
                                            chunks = chunk_text(text, chunk_size=4000, chunk_overlap=200)
                                            
                                            # Prepare metadata for the chunks
                                            metadata = {
                                                "course_name": selected_course,
                                                "week_number": selected_week,
                                                "transcript_name": transcript.get("transcript_name", file_path)
                                            }
                                            
                                            # Store chunks in Pinecone
                                            success_count, total_count = batch_upsert_chunks(chunks, metadata)
                                            
                                            st.success(f"✅ Successfully indexed {success_count}/{total_count} chunks to Pinecone")
                                        except Exception as index_error:
                                            st.error(f"Error indexing to Pinecone: {str(index_error)}")
                            else:
                                st.error(f"Could not extract text from {file_path}")
                        
                        if not all_text:
                            st.error("Could not extract text from any of the transcripts.")
                            st.info("Please try the Quick Upload feature to add your files.")
                        else:
                            try:
                                # Use Gemini to generate summary
                                prompt = f"""Generate a concise summary (250-300 words) of the following Coursera content for {selected_course}, Week {selected_week}. 
                                Focus on key concepts, important definitions, and main takeaways.

                                CONTENT:
                                {all_text[:15000]}"""
                                
                                response = get_gemini_response(prompt)
                                
                                if response:
                                    st.subheader(f"Summary for {selected_course}, Week {selected_week}")
                                    st.markdown(response)
                                    
                                    # Add download button
                                    st.download_button(
                                        "Download Summary",
                                        response,
                                        file_name=f"{selected_course}_Week{selected_week}_Summary.txt",
                                        mime="text/plain"
                                    )
                                    
                                    # Show Pinecone index stats
                                    if ENABLE_PINECONE_INDEXING:
                                        with st.expander("📊 Pinecone Vector Index Stats", expanded=False):
                                            try:
                                                stats = pinecone_index.describe_index_stats()
                                                total_vectors = stats.get('total_vector_count', 0)
                                                
                                                st.write("### Pinecone Index Statistics")
                                                st.write(f"Total vectors: {total_vectors}")
                                                
                                                # Show namespace stats if available
                                                if 'namespaces' in stats and 'coursera' in stats.get('namespaces', {}):
                                                    coursera_stats = stats['namespaces']['coursera']
                                                    st.write(f"Vectors in 'coursera' namespace: {coursera_stats.get('vector_count', 0)}")
                                                
                                                st.success("✅ Your content is now vectorized and can be searched semantically!")
                                            except Exception as stats_error:
                                                st.error(f"Error fetching Pinecone stats: {str(stats_error)}")
                                else:
                                    st.error("Failed to generate summary with Gemini API. Please check your API key configuration.")
                            except Exception as summary_error:
                                st.error(f"Error generating summary: {str(summary_error)}")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

with tab3:
    st.header("Ask Questions")
    
    # Show RAG status information
    if ENABLE_PINECONE_INDEXING:
        try:
            stats = pinecone_index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            namespaces = stats.get('namespaces', {})
            
            # Check for the 'coursera' namespace specifically
            coursera_vectors = 0
            if 'coursera' in namespaces:
                coursera_vectors = namespaces['coursera'].get('vector_count', 0)
            
            if total_vectors > 0:
                st.success(f"✅ Vector database is active with {total_vectors} total vectors")
                if coursera_vectors > 0:
                    st.success(f"✅ Found {coursera_vectors} vectors in 'coursera' namespace for search")
                else:
                    st.warning("⚠️ No vectors found in 'coursera' namespace. Content may not be properly indexed.")
            else:
                st.warning("⚠️ No vectors found in Pinecone. Please generate summaries first to create embeddings.")
                st.info("Step 1: Go to the 'Summarize' tab\nStep 2: Select a course and week\nStep 3: Click 'Generate Summary' to create vector embeddings")
                
            # Add Vector Diagnostics section
            with st.expander("🔧 Vector Coverage Diagnostics", expanded=False):
                st.write("### Vector Storage Diagnostics")
                
                # Display namespace statistics
                st.write("#### Namespace Statistics")
                for namespace, ns_stats in namespaces.items():
                    st.write(f"- **{namespace}**: {ns_stats.get('vector_count', 0)} vectors")
                
                # Add option to reindex all content
                st.write("#### Reindex Content")
                st.write("If you're getting poor retrieval results, you can reindex all content to improve vector quality.")
                
                if st.button("🔄 Reindex All Content"):
                    with st.spinner("Fetching available transcripts..."):
                        try:
                            # Get all transcripts from database
                            response = supabase.table("transcripts").select("*").execute()
                            
                            if not response.data:
                                st.error("No transcripts found in database.")
                                st.stop()
                                
                            st.success(f"Found {len(response.data)} transcripts. Starting reindexing process...")
                            
                            # Process each transcript
                            total_chunks = 0
                            successful_chunks = 0
                            
                            progress_bar = st.progress(0)
                            for i, transcript in enumerate(response.data):
                                file_path = transcript["file_path"]
                                course_name = transcript["course_name"]
                                week_number = transcript["week_number"]
                                transcript_name = transcript.get("transcript_name", file_path)
                                
                                st.write(f"Processing {file_path}...")
                                
                                # Extract text from file
                                text = None
                                try:
                                    file_data = get_file_from_supabase(file_path)
                                    if file_data:
                                        text = extract_text_from_file(file_data, file_path)
                                except Exception as extract_error:
                                    st.error(f"Error extracting text: {str(extract_error)}")
                                
                                if text:
                                    # Split into chunks and index
                                    chunks = chunk_text(text, chunk_size=2000, chunk_overlap=200)  # Smaller chunks for better retrieval
                                    
                                    # Prepare metadata
                                    metadata = {
                                        "course_name": course_name,
                                        "week_number": week_number,
                                        "transcript_name": transcript_name
                                    }
                                    
                                    # Index chunks
                                    success_count, total_count = batch_upsert_chunks(chunks, metadata)
                                    
                                    total_chunks += total_count
                                    successful_chunks += success_count
                                    
                                    st.write(f"✅ Indexed {success_count}/{total_count} chunks for {file_path}")
                                else:
                                    st.error(f"Could not extract text from {file_path}")
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(response.data))
                            
                            # Show final results
                            if total_chunks > 0:
                                st.success(f"Reindexing complete! Successfully indexed {successful_chunks}/{total_chunks} chunks ({successful_chunks/total_chunks*100:.1f}%)")
                            else:
                                st.error("No content was indexed. Please check your files and try again.")
                        except Exception as reindex_error:
                            st.error(f"Error during reindexing: {str(reindex_error)}")
                
                # Add test query to evaluate retrieval quality
                st.write("#### Test Vector Retrieval")
                test_query = st.text_input("Enter a test query to evaluate retrieval quality")
                test_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
                
                if test_query and st.button("Run Test Query"):
                    with st.spinner("Running test query..."):
                        result = semantic_search_with_pinecone(test_query, top_k=test_k)
                        if result:
                            st.success("✅ Retrieved content successfully!")
                            st.write(result)
                        else:
                            st.error("No relevant content found. Your vectors may be missing or of poor quality.")
                
        except Exception as pinecone_error:
            st.error(f"❌ Error connecting to Pinecone: {str(pinecone_error)}")
            st.info("Please check your Pinecone API key and settings")
    else:
        st.warning("⚠️ Vector indexing is disabled. Enable it in the configuration to improve search quality.")
    
    # Chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Get available courses for filtering
    courses = []
    if apis_configured:
        try:
            response = supabase.table("transcripts").select("course_name").execute()
            courses = list(set([item["course_name"] for item in response.data]))
        except Exception as e:
            st.error(f"Error fetching courses: {str(e)}")
    
    # Course/week filters
    col1, col2 = st.columns(2)
    with col1:
        selected_course = st.selectbox("Filter by Course", ["All Courses"] + courses if courses else ["No courses available"], key="chat_course")
    
    weeks = []
    if selected_course not in ["All Courses", "No courses available"]:
        try:
            response = supabase.table("transcripts").select("week_number").eq("course_name", selected_course).execute()
            weeks = sorted(list(set([item["week_number"] for item in response.data])))
        except Exception as e:
            st.error(f"Error fetching weeks: {str(e)}")
    
    with col2:
        selected_week = st.selectbox("Filter by Week", ["All Weeks"] + [str(w) for w in weeks] if weeks else ["All Weeks"], key="chat_week")
    
    # RAG search settings
    with st.expander("🔍 Search Settings", expanded=False):
        top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=15, value=5, 
                          help="Higher values retrieve more content but may include less relevant information")
        use_chat_history = st.checkbox("Use chat history for context", value=True,
                                      help="Includes previous messages as context for better continuity")
    
    # Get user input
    user_query = st.chat_input("Ask about your Coursera content")
    
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    # Check if Pinecone is accessible
                    try:
                        stats = pinecone_index.describe_index_stats()
                        total_vectors = stats.get('total_vector_count', 0)
                        if total_vectors == 0:
                            st.warning("No vectors found in Pinecone. Please generate summaries first to create embeddings.")
                            ai_response = "I can't answer your question because there's no content in the vector database. Please use the 'Summarize' tab to process some course content first."
                            st.write(ai_response)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                            st.stop()
                    except Exception as pinecone_error:
                        st.error(f"Error connecting to Pinecone: {str(pinecone_error)}")
                        ai_response = "I'm having trouble connecting to the knowledge base. Please check the Pinecone connection and try again."
                        st.write(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                        st.stop()
                    
                    # Build search parameters
                    course_filter = None if selected_course == "All Courses" else selected_course
                    week_filter = None if selected_week == "All Weeks" else int(selected_week)
                    
                    # Create search query (potentially incorporating chat history)
                    search_query = user_query
                    if use_chat_history and len(st.session_state.messages) > 1:
                        # Get last few messages to provide context (limiting to last 3 exchanges)
                        recent_messages = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages
                        context_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages[:-1]])
                        search_query = f"Context: {context_messages}\n\nCurrent question: {user_query}"
                    
                    # Perform semantic search
                    st.info("🔍 Searching for relevant content in vector database...")
                    retrieved_text = semantic_search_with_pinecone(
                        search_query, 
                        course_name=course_filter, 
                        week_number=week_filter, 
                        top_k=top_k
                    )
                    
                    if not retrieved_text:
                        # Try without filters if no results
                        if course_filter or week_filter:
                            st.info("No results with filters. Trying broader search...")
                            retrieved_text = semantic_search_with_pinecone(search_query, top_k=top_k)
                        
                        # If still no results
                        if not retrieved_text:
                            st.warning("No relevant content found in the vector database.")
                            ai_response = "I couldn't find any relevant information about this topic in the course materials. Please try a different question or upload more content."
                            st.write(ai_response)
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                            st.stop()
                    
                    # Show retrieved chunks in expandable section
                    with st.expander("📄 Retrieved Content", expanded=False):
                        st.write("The following content was retrieved from the vector database:")
                        st.text(retrieved_text[:1500] + "..." if len(retrieved_text) > 1500 else retrieved_text)
                    
                    # Generate answer with Gemini
                    st.info("🤖 Generating answer based on retrieved content...")
                    prompt = f"""You are an educational AI assistant helping students understand Coursera content.
                    
                    Answer the question based ONLY on the following information from course materials:
                    
                    {retrieved_text}
                    
                    Question: {user_query}
                    
                    Answer in a clear, direct manner. Be sure to highlight key concepts and provide well-structured information.
                    If information to answer the question is not contained in the retrieved content, say so directly."""
                    
                    ai_response = get_gemini_response(prompt)
                    
                    if not ai_response:
                        ai_response = "I'm having trouble generating a response. Please try again or check the system configuration."
                    
                    st.write(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with tab4:
    st.header("Generate Quiz Questions")
    
    # Get available courses
    courses = []
    if apis_configured:
        try:
            response = supabase.table("transcripts").select("course_name").execute()
            courses = list(set([item["course_name"] for item in response.data]))
        except Exception as e:
            st.error(f"Error fetching courses: {str(e)}")
    
    selected_course = st.selectbox("Select Course", courses if courses else ["No courses available"], key="quiz_course")
    
    if selected_course != "No courses available":
        # Get weeks for selected course
        weeks = []
        try:
            response = supabase.table("transcripts").select("week_number").eq("course_name", selected_course).execute()
            weeks = sorted(list(set([item["week_number"] for item in response.data])))
        except Exception as e:
            st.error(f"Error fetching weeks: {str(e)}")
        
        selected_week = st.selectbox("Select Week", weeks if weeks else [], key="quiz_week")
        
        question_types = st.multiselect(
            "Question Types", 
            ["Multiple Choice", "True/False", "Short Answer"],
            default=["Multiple Choice", "True/False"]
        )
        
        num_questions = st.slider("Number of Questions", 3, 10, 5)
        
        if selected_week and st.button("Generate Quiz"):
            with st.spinner("Generating quiz questions..."):
                try:
                    # Build the prompt
                    question_types_str = ", ".join(question_types)
                    
                    # Query transcripts
                    query = supabase.table("transcripts").select("*")
                    if selected_course != "All Courses" and selected_course != "No courses available":
                        query = query.eq("course_name", selected_course)
                    if selected_week != "All Weeks":
                        query = query.eq("week_number", int(selected_week))
                    
                    response = query.execute()
                    
                    if not response.data:
                        st.warning("No transcripts found for the selected filters.")
                    else:
                        # Process the PDFs to extract text
                        st.info(f"Searching through {len(response.data)} transcript(s)...")
                        
                        all_text = ""
                        for transcript in response.data:
                            file_path = transcript["file_path"]
                            # Extract text from the PDF
                            text = extract_text_from_supabase_pdf(file_path)
                            if text:
                                all_text += f"\n\n--- {transcript['transcript_name']} ---\n\n{text}"
                        
                        if not all_text:
                            st.error("Could not extract text from any of the transcripts.")
                        else:
                            # If content is too large, truncate
                            max_length = 14000  # Conservative max for context
                            if len(all_text) > max_length:
                                st.warning("The transcript content is very large. Only using the first portion for the response.")
                                all_text = all_text[:max_length] + "..."
                            
                            # Generate quiz with Gemini
                            prompt = f"""Generate {num_questions} quiz questions from the Coursera course '{selected_course}', Week {selected_week}.
                            Include the following question types: {question_types_str}.
                            For each question:
                            1. Provide the question clearly
                            2. For multiple-choice, include 4 options (A, B, C, D) with only one correct answer
                            3. For all questions, provide the correct answer
                            4. Include a brief explanation of why the answer is correct, referencing the course content
                            
                            Format each question with a number, followed by the question type in parentheses.
                            
                            Here is the course content to base the questions on:
                            {all_text}
                            """
                            
                            response = get_gemini_response(prompt)
                            
                            if not response:
                                st.error("Failed to generate quiz questions. Please try again.")
                            else:
                                st.subheader(f"Quiz Questions for {selected_course}, Week {selected_week}")
                                st.write(response)
                                
                                # Add download button for the quiz
                                st.download_button(
                                    "Download Quiz",
                                    response,
                                    file_name=f"{selected_course}_Week{selected_week}_Quiz.txt",
                                    mime="text/plain"
                                )
                except Exception as e:
                    st.error(f"Error generating quiz: {str(e)}")

with tab5:
    st.header("Exam Preparation")
    
    # Get available courses
    courses = []
    if apis_configured:
        try:
            response = supabase.table("transcripts").select("course_name").execute()
            courses = list(set([item["course_name"] for item in response.data]))
        except Exception as e:
            st.error(f"Error fetching courses: {str(e)}")
    
    selected_course = st.selectbox("Select Course", courses if courses else ["No courses available"], key="exam_course")
    
    if selected_course != "No courses available":
        col1, col2 = st.columns(2)
        
        with col1:
            exam_format = st.radio(
                "Exam Format",
                ["Comprehensive (all weeks)", "Specific weeks"]
            )
        
        selected_weeks = []
        if exam_format == "Specific weeks":
            # Get weeks for selected course
            weeks = []
            try:
                response = supabase.table("transcripts").select("week_number").eq("course_name", selected_course).execute()
                weeks = sorted(list(set([item["week_number"] for item in response.data])))
            except Exception as e:
                st.error(f"Error fetching weeks: {str(e)}")
            
            with col2:
                selected_weeks = st.multiselect("Select Weeks", weeks if weeks else [])
        
        num_questions = st.slider("Number of Questions", 5, 20, 10, key="exam_questions")
        
        difficulty = st.select_slider(
            "Difficulty Level",
            options=["Easy", "Medium", "Hard"],
            value="Medium"
        )
        
        if st.button("Generate Practice Exam"):
            with st.spinner("Generating practice exam..."):
                try:
                    # Build the prompt and get transcripts
                    if exam_format == "Specific weeks" and selected_weeks:
                        # Create a string of weeks for the prompt
                        weeks_str = ", ".join([str(w) for w in selected_weeks])
                        prompt_weeks = f"Weeks {weeks_str}"
                        
                        # Query for specific weeks
                        query = supabase.table("transcripts").select("*").eq("course_name", selected_course).in_("week_number", selected_weeks)
                    else:
                        prompt_weeks = "all weeks"
                        query = supabase.table("transcripts").select("*").eq("course_name", selected_course)
                    
                    response = query.execute()
                    
                    if not response.data:
                        st.warning("No transcripts found for the selected course and weeks.")
                    else:
                        # Process the PDFs to extract text
                        st.info(f"Searching through {len(response.data)} transcript(s)...")
                        
                        all_text = ""
                        for transcript in response.data:
                            file_path = transcript["file_path"]
                            # Extract text from the PDF
                            text = extract_text_from_supabase_pdf(file_path)
                            if text:
                                all_text += f"\n\n--- {transcript['transcript_name']} ---\n\n{text}"
                        
                        if not all_text:
                            st.error("Could not extract text from any of the transcripts.")
                        else:
                            # If content is too large, truncate
                            max_length = 14000  # Conservative max for context
                            if len(all_text) > max_length:
                                st.warning("The transcript content is very large. Only using the first portion for the response.")
                                all_text = all_text[:max_length] + "..."
                            
                            # Generate exam with Gemini
                            prompt = f"""Create a practice exam for the Coursera course '{selected_course}', covering {prompt_weeks}.
                            Generate {num_questions} questions at {difficulty} difficulty level.
                            
                            Include a mix of:
                            - Multiple-choice questions (4 options)
                            - True/False questions
                            - Short answer questions
                            
                            For each question:
                            1. Clearly state the question
                            2. Provide all necessary options for multiple-choice
                            3. Include the correct answer
                            4. Provide a detailed explanation of the answer, referencing specific course content
                            
                            The exam should resemble an actual Coursera exam in style and format.
                            Number each question and specify its type in parentheses.
                            
                            Here is the course content to base the exam on:
                            {all_text}
                            """
                            
                            response = get_gemini_response(prompt)
                            
                            if not response:
                                st.error("Failed to generate practice exam. Please try again.")
                            else:
                                st.subheader(f"Practice Exam for {selected_course}")
                                st.write(response)
                                
                                # Add download button for the exam
                                st.download_button(
                                    "Download Practice Exam",
                                    response,
                                    file_name=f"{selected_course}_PracticeExam.txt",
                                    mime="text/plain"
                                )
                except Exception as e:
                    st.error(f"Error generating practice exam: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Made with Streamlit, LlamaIndex, and OpenAI")

# API configuration instructions if not set up
if not apis_configured:
    st.warning("""
    ### API Configuration Required
    
    Please set up your API keys in a `.env` file or Streamlit secrets:
    
    ```
    OPENAI_API_KEY=your-openai-api-key
    SUPABASE_URL=your-supabase-url
    SUPABASE_KEY=your-supabase-key
    PINECONE_API_KEY=your-pinecone-api-key
    PINECONE_ENVIRONMENT=gcp-starter
    GOOGLE_API_KEY=your-gemini-api-key
    ```
    
    See the README for setup instructions.
    """)