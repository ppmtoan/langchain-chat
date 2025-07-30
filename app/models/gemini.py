import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.config import Config

class GeminiModel:
    def __init__(self):
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())
            self.llm = ChatGoogleGenerativeAI(
                model=Config.GEMINI_MODEL,
                temperature=Config.TEMPERATURE,
                max_tokens=Config.MAX_TOKENS,
                timeout=Config.TIMEOUT,
                max_retries=Config.MAX_RETRIES
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(model=Config.EMBEDDING_MODEL)
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            self.llm = None
            self.embeddings = None

    def invoke(self, messages):
        if self.llm:
            return self.llm.invoke(messages)
        raise Exception("Gemini model not initialized")