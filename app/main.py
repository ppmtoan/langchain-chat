import sys
import os
import uuid
import streamlit as st
from app.config import Config
from app.database.supabase_client import SupabaseClient
from app.database.vector_store import VectorStore
from app.models.gemini import GeminiModel
from app.utils.image_handler import process_image
from app.utils.document_loader import DocumentLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain import hub

# Add current working directory to Python path
sys.path.append(os.getcwd())

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "document_ids" not in st.session_state:
        st.session_state.document_ids = []

def load_css():
    """Load custom CSS styles."""
    with open("app/templates/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def setup_api_keys():
    """Configure API keys from secrets or user input."""
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets:
        st.subheader("Enter API Keys")
        supabase_url = st.text_input("Supabase URL:", type="default")
        supabase_key = st.text_input("Supabase Service Role Key:", type="password")
        google_api_key = st.text_input("Google AI API Key:", type="password")
        if supabase_url and supabase_key and google_api_key:
            st.session_state["SUPABASE_URL"] = supabase_url
            st.session_state["SUPABASE_KEY"] = supabase_key
            st.session_state["GOOGLE_API_KEY"] = google_api_key
            st.success("API keys set successfully!")
            return False
    else:
        os.environ[Config.SUPABASE_URL] = st.secrets["SUPABASE_URL"]
        os.environ[Config.SUPABASE_KEY] = st.secrets["SUPABASE_KEY"]
        os.environ[Config.GOOGLE_API_KEY] = st.secrets["GOOGLE_API_KEY"]
        return True
    return False

def initialize_components():
    """Initialize core components."""
    document_loader = DocumentLoader()
    supabase_client = SupabaseClient()
    gemini_model = GeminiModel()
    vector_store = VectorStore(supabase_client.client, gemini_model.embeddings)
    return document_loader, supabase_client, gemini_model, vector_store

def handle_document_upload(document_loader, supabase_client, vector_store):
    """Handle document upload and processing."""
    st.subheader("Upload Document (PDF or TXT)")
    uploaded_doc = st.file_uploader("Upload a document (optional)", type=["pdf", "txt"])
    if uploaded_doc:
        try:
            documents, document_id = document_loader.load_document(uploaded_doc, supabase_client.client, st.session_state.session_id)
            if documents:
                vector_store.add_documents(documents, st.session_state.session_id, document_id)
                st.session_state.document_ids.append(document_id)
                st.success(f"Document {uploaded_doc.name} loaded successfully!")
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")

def handle_image_upload(supabase_client):
    """Handle image upload and processing."""
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
    return process_image(uploaded_file, supabase_client.client) if uploaded_file else (None, None)

def process_user_input(user_input, image_url, supabase_client, gemini_model, vector_store, system_prompt):
    """Process user input and generate response."""
    if user_input or image_url:
        content = [{"type": "text", "text": user_input}] if user_input else []
        if image_url:
            content.append({"type": "image_url", "image_url": image_url})

        user_message = {"role": "user", "content": user_input or "Image provided", "image_url": image_url}
        st.session_state.messages.append(user_message)
        supabase_client.save_message(st.session_state.session_id, "user", user_input or "Image provided", image_url)

        messages = [system_prompt]
        context = vector_store.get_context(user_input or "Describe the image")
        if context:
            context_text = "\n".join([doc.page_content for doc in context])
            messages.append(SystemMessage(content=f"Context from conversation and documents: {context_text}"))
        messages.append(HumanMessage(content=content))

        try:
            response = gemini_model.invoke(messages)
            assistant_message = {"role": "assistant", "content": response.content, "image_url": None}
            st.session_state.messages.append(assistant_message)
            supabase_client.save_message(st.session_state.session_id, "assistant", response.content)
            vector_store.initialize(st.session_state.messages, st.session_state.document_ids, st.session_state.session_id)
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")
    else:
        st.warning("Please enter a message, upload an image, or upload a document.")

def display_conversation_history():
    """Display the conversation history."""
    st.subheader("Conversation History")
    for msg in st.session_state.messages:
        with st.container():
            if msg["role"] == "user":
                st.markdown(f"**You**: {msg['content']}")
                if msg.get("image_url"):
                    st.image(msg["image_url"], width=200)
            else:
                st.markdown(f"**Assistant**: {msg['content']}")

def clear_chat_history(supabase_client, vector_store):
    """Clear chat history and reset session."""
    if st.button("Clear Chat History"):
        try:
            supabase_client.clear_history(st.session_state.session_id)
            vector_store.clear(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.document_ids = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("Chat history and documents cleared!")
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")

def main():
    """Main application function."""
    st.set_page_config(page_title="Gemini Chat App with Supabase", layout="wide")
    st.title("Gemini Chat App with Supabase")
    
    initialize_session_state()
    load_css()

    if not setup_api_keys():
        st.warning("Please enter all required API keys to continue.")
        return

    document_loader, supabase_client, gemini_model, vector_store = initialize_components()
    
    if not supabase_client.client or not gemini_model.llm or not gemini_model.embeddings:
        return

    st.session_state.messages = supabase_client.load_messages(st.session_state.session_id)
    
    handle_document_upload(document_loader, supabase_client, vector_store)
    
    system_prompt = SystemMessage(content="You are a helpful assistant that can process text, images, and documents. Provide accurate and concise responses, using provided document context when relevant.")
    
    image_path, image_url = handle_image_upload(supabase_client)
    
    user_input = st.text_area("Your message:", height=100)
    
    if st.button("Send"):
        process_user_input(user_input, image_url, supabase_client, gemini_model, vector_store, system_prompt)
    
    display_conversation_history()
    clear_chat_history(supabase_client, vector_store)

if __name__ == "__main__":
    main()