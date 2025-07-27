# Gemini Chat App with Supabase

A chat application built with Streamlit, LangChain, Google Gemini, and Supabase for persistent storage and vector search.

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd chat-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in your API keys.
   - Get Google AI API key from https://ai.google.dev/gemini-api/docs/api-key
   - Get Supabase URL and Service Role Key from your Supabase project dashboard.

4. Set up Supabase:
   - Enable `pgvector` extension in Supabase SQL Editor (Dashboard > SQL Editor > Quick Start > LangChain).
   - Run `scripts/setup_supabase.sql` in the Supabase SQL Editor to create the `messages` table.

5. Temporarily add the current directory to the Python path:

   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

6. Run the app:
   ```bash
   ./run.sh
   ```

## Features
- Text and image input support
- Persistent conversation history in Supabase
- Vector search for context-aware responses
- Custom-styled Streamlit UI
- Session management for multiple users

## Testing
Run tests with:
```bash
python -m unittest discover tests