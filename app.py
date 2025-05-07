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

# Load environment variables
load_dotenv()

# Configuration flags - ADD THIS SECTION
ENABLE_PINECONE_INDEXING = False  # Set to False to disable vector indexing
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
    supabase_url = os.environ.get("SUPABASE_URL") 
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase_service_key = os.environ.get("SUPABASE_SERVICE_KEY")
    
    # If any of the keys are missing, try to get them from secrets
    if not supabase_url or not supabase_key or not supabase_service_key:
        supabase_url = st.secrets.get("SUPABASE_URL", supabase_url)
        supabase_key = st.secrets.get("SUPABASE_KEY", supabase_key)
        supabase_service_key = st.secrets.get("SUPABASE_SERVICE_KEY", supabase_service_key)
    
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
        supabase_url = os.environ.get("SUPABASE_URL") 
        supabase_key = os.environ.get("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            supabase_url = st.secrets.get("SUPABASE_URL")
            supabase_key = st.secrets.get("SUPABASE_KEY")
        
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

# Modify the get_file_from_supabase function to be more resilient
def get_file_from_supabase(file_path):
    """Download a file from Supabase storage and return its contents"""
    global supabase
    
    # Sanitize file path - remove any whitespace or newlines
    clean_path = file_path.strip()
    
    # Skip if empty path
    if not clean_path:
        return None
        
    # Try to get the file from Supabase
    for attempt in range(2):  # Make up to 2 attempts
        try:
            # Ensure connection is valid
            if attempt > 0:
                verify_supabase_connection()
                
            # Get direct info about the bucket
            try:
                all_buckets = supabase.storage.list_buckets()
                bucket_names = [b.get('name') for b in all_buckets]
                if 'transcripts' not in bucket_names:
                    st.error("'transcripts' bucket not found in your Supabase project!")
                    st.write(f"Available buckets: {', '.join(bucket_names)}")
                    break
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
                        st.info("If one of these is the file you want, update your database record.")
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
    
    return None

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
        
        # Add the chunk
        chunks.append(text[start:end])
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
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        with st.spinner(f"Processing batch {i//batch_size + 1} of {(len(chunks)-1)//batch_size + 1}..."):
            for chunk in batch:
                success, message = upsert_to_pinecone(chunk, metadata)
                if success:
                    successful_upserts += 1
                    
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
        vector_id = f"{metadata['course_name']}_{metadata['week_number']}_{uuid.uuid4()}"
        
        # Check if the index supports integrated embeddings (field_map)
        try:
            index_desc = pinecone_index.describe_index_stats()
            if hasattr(index_desc, 'dimension'):
                # This is a standard vector index, we need to generate embeddings ourselves
                embed_model = OpenAIEmbedding()
                
                # Retry logic for API connection errors
                max_retries = 3
                retry_count = 0
                embedding = None
                
                while retry_count < max_retries:
                    try:
                        embedding = embed_model.get_text_embedding(text)
                        break
                    except Exception as e:
                        if "APIConnectionError" in str(e) or "Connection error" in str(e):
                            retry_count += 1
                            if retry_count < max_retries:
                                wait_time = 1 * (2 ** retry_count)  # Exponential backoff
                                st.warning(f"API connection error, retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                st.error("Maximum retries reached for embedding API")
                                return False, f"Embedding API connection error after {max_retries} retries"
                        else:
                            raise
                
                if embedding:
                    # Upsert the vector
                    pinecone_index.upsert(
                        vectors=[(
                            vector_id, 
                            embedding, 
                            {
                                "text": text[:1000],  # Store first 1000 chars of text
                                "course_name": metadata["course_name"],
                                "week_number": metadata["week_number"],
                                "transcript_name": metadata["transcript_name"]
                            }
                        )],
                        namespace="coursera"
                    )
                    return True, f"Indexed content with ID: {vector_id}"
                else:
                    return False, "Failed to generate embedding"
            else:
                # This might be an index with integrated embedding
                try:
                    # Use the newer upsert_records which can automatically convert text to vectors
                    pinecone_index.upsert_records(
                        namespace="coursera",
                        records=[{
                            "_id": vector_id,
                            "chunk_text": text,  # Will be converted to vector automatically
                            "course_name": metadata["course_name"],
                            "week_number": metadata["week_number"],
                            "transcript_name": metadata["transcript_name"]
                        }]
                    )
                    return True, f"Indexed text with ID: {vector_id} using integrated embedding"
                except Exception as e:
                    if "no text embedding field_map" in str(e).lower():
                        st.warning("Index doesn't support integrated embedding. Falling back to manual embedding.")
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
                    {
                        "text": text[:1000],  # Store first 1000 chars of text
                        "course_name": metadata["course_name"],
                        "week_number": metadata["week_number"],
                        "transcript_name": metadata["transcript_name"]
                    }
                )],
                namespace="coursera"
            )
            return True, f"Indexed content with ID: {vector_id} (fallback method)"
            
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
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    retrieved_texts.append(match.metadata['text'])
            
            if not retrieved_texts:
                st.warning("No relevant vectors found in Pinecone. This may happen if you haven't indexed this content before.")
                return ""
                
            return "\n\n".join(retrieved_texts)
        except Exception as query_error:
            st.warning(f"Error in Pinecone query: {str(query_error)}")
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
                            
                    if filtered_texts:
                        return "\n\n".join(filtered_texts)
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
    
    # Add an expanded credentials section in the Check Supabase Storage expander
    with st.expander("🔍 Check Supabase Storage", expanded=True):
        st.write("### Supabase Credentials")
        st.info("Your Supabase buckets appear to be inaccessible. Let's verify your credentials.")
        
        # Show current credentials (masked)
        supabase_url = os.environ.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
        supabase_key = os.environ.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
        
        # Mask credentials for display
        masked_url = supabase_url[:20] + "..." + supabase_url[-10:] if len(supabase_url) > 30 else supabase_url
        masked_key = supabase_key[:5] + "..." + supabase_key[-5:] if len(supabase_key) > 10 else supabase_key
        
        st.write(f"**Current URL:** `{masked_url}`")
        st.write(f"**Current Key:** `{masked_key}`")
        
        # Add manual credential entry
        st.write("### Update Credentials")
        st.write("You can temporarily update your Supabase credentials here:")
        
        # Save credentials in session state if they don't exist
        if "manual_supabase_url" not in st.session_state:
            st.session_state.manual_supabase_url = supabase_url
        if "manual_supabase_key" not in st.session_state:
            st.session_state.manual_supabase_key = supabase_key
            
        # Input fields for credentials
        new_url = st.text_input("Supabase URL", 
                               value=st.session_state.manual_supabase_url,
                               help="Find this in your Supabase dashboard under Project Settings > API",
                               key="input_supabase_url")
        new_key = st.text_input("Supabase API Key", 
                               value=st.session_state.manual_supabase_key,
                               help="Use the 'anon' key from Project Settings > API",
                               type="password",
                               key="input_supabase_key")
        
        # Update button
        if st.button("Update Credentials"):
            try:
                # Use a temporary variable first, only update global after successful test
                st.session_state.manual_supabase_url = new_url
                st.session_state.manual_supabase_key = new_key
                
                # Try to create a new client with these credentials
                temp_client = create_client(new_url, new_key)
                
                # Test with a simple operation
                test_result = temp_client.auth.get_session()
                
                # If successful, update the client
                supabase = temp_client
                
                st.success("✅ Credentials updated and connection successful!")
                
                # Try to list buckets with new credentials
                try:
                    buckets = supabase.storage.list_buckets()
                    if buckets:
                        st.success(f"Found {len(buckets)} buckets: {[b.get('name') for b in buckets]}")
                    else:
                        st.warning("Connection successful but no buckets found.")
                except Exception as bucket_error:
                    st.error(f"Connection successful but bucket access failed: {str(bucket_error)}")
            except Exception as e:
                st.error(f"Failed to connect with new credentials: {str(e)}")
        
        # Add direct API test button
        if st.button("Test Direct API Access"):
            try:
                # Use current credentials (either from session state or environment)
                current_url = st.session_state.get("manual_supabase_url") or supabase_url
                current_key = st.session_state.get("manual_supabase_key") or supabase_key
                
                if not current_url or not current_key:
                    st.error("Missing URL or API key")
                else:
                    # Remove any trailing slashes from URL
                    api_url = current_url.rstrip("/")
                    
                    # Create headers for requests
                    headers = {
                        "apikey": current_key,
                        "Authorization": f"Bearer {current_key}"
                    }
                    
                    # Use httpx for direct API calls
                    import httpx
                    
                    # Test connection to Supabase REST API
                    st.write("Testing REST API connection...")
                    with httpx.Client(headers=headers) as client:
                        # Try to access storage API
                        storage_url = f"{api_url}/storage/v1/bucket"
                        storage_response = client.get(storage_url)
                        
                        if storage_response.status_code == 200:
                            buckets = storage_response.json()
                            st.success(f"✅ Direct storage API access successful! Found {len(buckets)} buckets")
                            st.json(buckets)
                            
                            # If transcripts bucket exists, try to list files
                            transcript_bucket = next((b for b in buckets if b.get('name') == 'transcripts'), None)
                            if transcript_bucket:
                                files_url = f"{api_url}/storage/v1/object/list/transcripts"
                                files_response = client.get(files_url)
                                
                                if files_response.status_code == 200:
                                    files = files_response.json()
                                    st.success(f"✅ Found {len(files)} files in the transcripts bucket!")
                                    # Create a file list table
                                    if files:
                                        file_table = []
                                        for file in files:
                                            file_table.append({
                                                "Name": file.get("name", ""),
                                                "Size": f"{int(file.get('metadata', {}).get('size', 0) / 1024)} KB",
                                                "Last Modified": file.get("created_at", "")
                                            })
                                        st.table(file_table)
                                else:
                                    st.error(f"❌ Failed to list files: {files_response.status_code} - {files_response.text}")
                        else:
                            st.error(f"❌ Storage API access failed: {storage_response.status_code} - {storage_response.text}")
            except Exception as api_error:
                st.error(f"Direct API test failed: {str(api_error)}")
                import traceback
                st.code(traceback.format_exc())
        
        # Add the standard diagnostic buttons in a row
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("Test Connection"):
                with st.spinner("Testing Supabase connection..."):
                    if verify_supabase_connection():
                        st.success("✅ Supabase connection is working")
                    else:
                        st.error("❌ Supabase connection failed")
        
        with col2:
            if st.button("List Buckets"):
                try:
                    buckets = supabase.storage.list_buckets()
                    if buckets:
                        st.success(f"Found {len(buckets)} buckets")
                        st.write([bucket.get('name') for bucket in buckets])
                    else:
                        st.warning("No buckets found")
                except Exception as e:
                    st.error(f"Error listing buckets: {str(e)}")
        
        with col3:
            if st.button("List Files"):
                try:
                    files = supabase.storage.from_("transcripts").list()
                    if files:
                        st.success(f"Found {len(files)} files in 'transcripts' bucket")
                        # Display file list in a table
                        file_data = []
                        for file in files:
                            file_data.append({
                                "Name": file.get("name", "Unknown"),
                                "Size": f"{int(file.get('metadata', {}).get('size', 0) / 1024)} KB",
                                "Last Modified": file.get("created_at", "Unknown")
                            })
                        st.table(file_data)
                    else:
                        st.warning("No files found in 'transcripts' bucket")
                except Exception as e:
                    st.error(f"Error listing files: {str(e)}")
                
        # Continue with existing Quick Upload functionality
        st.write("### Quick Upload")
        upload_course = st.text_input("Course Name", placeholder="e.g., AI For Everyone", key="quick_upload_course")
        upload_week = st.number_input("Week Number", min_value=1, max_value=20, value=1, key="quick_upload_week")
        upload_file = st.file_uploader("Upload File", type=["pdf", "txt", "json", "md"], key="quick_upload_file")
        
        if upload_file and upload_course and st.button("Upload to Supabase"):
            try:
                # Generate clean file name
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
                            
                            # Extract text from the PDF
                            text = extract_text_from_supabase_pdf(file_path)
                            if text:
                                all_text += text + "\n\n"
                                
                                # Check if this specific transcript is already indexed
                                if not check_if_already_indexed(selected_course, selected_week, transcript["transcript_name"]):
                                    # Only index if not already present
                                    st.info(f"Indexing new content from {file_path}...")
                                    # Index this content using our direct function instead of LlamaIndex
                                    chunks = chunk_text(text, chunk_size=1000, chunk_overlap=200)
                                    
                                    # Process in batches
                                    successful, total = batch_upsert_chunks(
                                        chunks,
                                        {
                                            "course_name": selected_course,
                                            "week_number": selected_week,
                                            "transcript_name": transcript["transcript_name"]
                                        },
                                        batch_size=20
                                    )
                                    
                                    if successful > 0:
                                        st.success(f"Indexed {successful}/{total} chunks from {file_path}")
                                    else:
                                        st.warning(f"Failed to index content from {file_path}")
                                else:
                                    st.info(f"Content from {file_path} already indexed, skipping...")
                            else:
                                st.warning(f"Could not extract text from {file_path}")
                        
                        if all_text:
                            try:
                                # Check if summary content is already indexed
                                if not check_if_already_indexed(selected_course, selected_week, "summary"):
                                    # First, ensure the content is indexed
                                    chunks = chunk_text(all_text, chunk_size=1000, chunk_overlap=200)
                                    
                                    with st.spinner("Indexing content for retrieval..."):
                                        # Process chunks in batches
                                        successful, total = batch_upsert_chunks(
                                            # Limit to 30 chunks to avoid overwhelming the API
                                            chunks[:min(30, len(chunks))],
                                            {
                                                "course_name": selected_course,
                                                "week_number": selected_week,
                                                "transcript_name": "summary"
                                            },
                                            batch_size=10  # Smaller batch size for more frequent updates
                                        )
                                        
                                        if successful > 0:
                                            st.success(f"Successfully indexed {successful}/{min(30, total)} chunks")
                                        else:
                                            st.warning("Could not index any chunks, will use direct text processing")
                                else:
                                    st.info("Summary content already indexed, using existing vectors...")
                                
                                # Now perform semantic search for relevant content
                                with st.spinner("Performing semantic search for relevant concepts..."):
                                    summary_query = f"key concepts and main takeaways from {selected_course} week {selected_week}"
                                    relevant_content = ""
                                    
                                    try:
                                        relevant_content = semantic_search_with_pinecone(
                                            summary_query, 
                                            course_name=selected_course, 
                                            week_number=selected_week,
                                            top_k=5
                                        )
                                        
                                        if relevant_content:
                                            st.info(f"Found {len(relevant_content.split())} words of relevant content")
                                    except Exception as search_error:
                                        st.warning(f"Semantic search error: {str(search_error)}. Using full content instead.")
                                        relevant_content = ""
                                
                                # If we got relevant content, use it for the summary
                                context_for_summary = relevant_content if relevant_content else all_text[:12000]
                                
                                # Generate summary with Gemini
                                prompt = f"""Generate a concise summary (250-300 words) of the following Coursera content for {selected_course}, Week {selected_week}. 
                                Focus on key concepts, important definitions, and main takeaways.

                                CONTENT:
                                {context_for_summary}"""
                                
                                response = get_gemini_response(prompt)
                                
                                if response:
                                    st.subheader(f"Summary for {selected_course}, Week {selected_week}")
                                    st.write(response)
                                    
                                    # Add download button
                                    st.download_button(
                                        "Download Summary",
                                        response,
                                        file_name=f"{selected_course}_Week{selected_week}_Summary.txt",
                                        mime="text/plain"
                                    )
                                else:
                                    st.error("Failed to generate summary with Gemini")
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                                # Fallback to simple summarization
                                try:
                                    prompt = f"""Generate a concise summary (250-300 words) of the following Coursera content for {selected_course}, Week {selected_week}. 
                                    Focus on key concepts, important definitions, and main takeaways.
                                    
                                    CONTENT:
                                    {all_text[:10000]}"""
                                    
                                    response = get_gemini_response(prompt)
                                    if response:
                                        st.subheader(f"Summary for {selected_course}, Week {selected_week} (Fallback Method)")
                                        st.write(response)
                                except Exception as fallback_error:
                                    st.error(f"Even fallback summarization failed: {str(fallback_error)}")
                        else:
                            st.error("Could not extract text from any of the transcripts.")
                except Exception as e:
                    st.error(f"Error generating summary: {str(e)}")

with tab3:
    st.header("Ask Questions")
    
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
    
    # Get user input
    user_query = st.chat_input("Ask about your Coursera content")
    
    if user_query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build query to get relevant files
                    query = supabase.table("transcripts").select("*")
                    
                    if selected_course != "All Courses" and selected_course != "No courses available":
                        query = query.eq("course_name", selected_course)
                    
                    if selected_week != "All Weeks":
                        query = query.eq("week_number", int(selected_week))
                    
                    response = query.execute()
                    
                    if not response.data:
                        st.warning("No transcripts found for the selected filters.")
                        ai_response = "I don't have any information for your query. Please check if you have uploaded relevant transcripts and selected the correct course/week."
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
                            ai_response = "I couldn't extract text from any of the transcripts. Please check if the PDFs are valid and try again."
                        else:
                            # If content is too large, truncate
                            max_length = 14000  # Conservative max for context
                            if len(all_text) > max_length:
                                st.warning("The transcript content is very large. Only using the first portion for the response.")
                                all_text = all_text[:max_length] + "..."
                            
                            # Use Gemini to generate response
                            prompt = f"Here is the transcript content:\n\n{all_text}\n\nBased only on this content, please answer the following question:\n{user_query}"
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