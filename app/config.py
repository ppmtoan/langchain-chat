#### `app/config.py`
class Config:
    SUPABASE_URL = "SUPABASE_URL"
    SUPABASE_KEY = "SUPABASE_KEY"
    GOOGLE_API_KEY = "GOOGLE_API_KEY"
    GEMINI_MODEL = "gemini-2.0-flash"
    EMBEDDING_MODEL = "models/gemini-embedding-exp-03-07"
    MAX_TOKENS = 1000
    TEMPERATURE = 0.7
    TIMEOUT = 30
    MAX_RETRIES = 2