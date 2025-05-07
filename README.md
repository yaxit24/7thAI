# Coursera Study Buddy

A Streamlit application that helps students interact with their Coursera course content using AI. Upload transcripts, generate summaries, ask questions, create quizzes, and prepare for exams.

> **Note**: This project has been optimized for deployment with unnecessary test files removed and configuration updated for both Streamlit Cloud and Vercel deployment options.

## Features

- **Upload Transcripts**: Upload PDF transcripts from Coursera courses
- **Summarize Content**: Generate concise summaries of course materials
- **Ask Questions**: Get answers to questions based on the course content
- **Generate Quizzes**: Create quiz questions from the course material
- **Exam Preparation**: Generate practice exams with varying difficulty levels
- **AI Models**: Supports both OpenAI GPT and Google Gemini models

## Prerequisites

- Python 3.9+
- Streamlit
- OpenAI API key
- Google Gemini API key (optional)
- Supabase account (for storage and database)
- Pinecone account (for vector search)

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd coursera-study-buddy
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up API keys

Create a `.streamlit/secrets.toml` file with your API keys:

```toml
OPENAI_API_KEY = "your-openai-api-key"
GOOGLE_API_KEY = "your-gemini-api-key"  # Optional
SUPABASE_URL = "your-supabase-url"
SUPABASE_KEY = "your-supabase-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_ENVIRONMENT = "gcp-starter"
```

Alternatively, you can set these as environment variables.

### 5. Supabase Setup

1. Create a new Supabase project
2. Set up a storage bucket named "transcripts"
3. Create appropriate storage policies to allow file uploads
4. The application will automatically create the necessary tables

### 6. Pinecone Setup

1. Create a Pinecone account
2. Create a serverless index with the name "coursera-transcripts"
3. Use dimension 1536 (for OpenAI embeddings) and cosine similarity
4. Make sure your environment matches the one in your config

## Running the application

```bash
streamlit run app.py
```

## Deployment

### Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set the main file path to `app.py`
5. Add your secrets in the Streamlit Cloud dashboard

### Deploying to Vercel

This repository includes configuration for Vercel deployment:

1. Install Vercel CLI: `npm install -g vercel`
2. Run `vercel` in the project directory
3. Set environment variables in the Vercel dashboard

## Usage

1. **Upload**: Start by uploading PDF transcripts from your Coursera courses
2. **Summarize**: Generate concise summaries of your course materials
3. **Ask Questions**: Use the chat interface to ask questions about the content
4. **Quiz**: Generate quiz questions to test your knowledge
5. **Exam Prep**: Create practice exams to prepare for your actual course exams

## Troubleshooting

- If you encounter storage issues, verify your Supabase bucket permissions
- For vector search issues, check your Pinecone index configuration
- API key errors will appear in the streamlit logs

## License

[MIT](LICENSE)
# 7thAI
