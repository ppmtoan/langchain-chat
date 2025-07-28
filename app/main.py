import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from app.config import Config
from app.database.supabase_client import SupabaseClient
from app.database.vector_store import VectorStore
from app.models.gemini import GeminiModel
from app.utils.image_handler import process_image
from langchain_core.messages import HumanMessage, SystemMessage
import uuid
import os

# Streamlit page configuration
st.set_page_config(page_title="Gemini Chat App with Supabase", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Load CSS
with open("app/templates/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def setup_api_keys():
    if not all(key in os.environ for key in [Config.SUPABASE_URL, Config.SUPABASE_KEY, Config.GOOGLE_API_KEY]):
        st.subheader("Enter API Keys")
        supabase_url = st.text_input("Supabase URL:", type="default")
        supabase_key = st.text_input("Supabase Service Role Key:", type="password")
        google_api_key = st.text_input("Google AI API Key:", type="password")
        if supabase_url and supabase_key and google_api_key:
            os.environ[Config.SUPABASE_URL] = supabase_url
            os.environ[Config.SUPABASE_KEY] = supabase_key
            os.environ[Config.GOOGLE_API_KEY] = google_api_key
            st.success("API keys set successfully!")
        return False
    return True

def main():
    st.title("Gemini Chat App with Supabase")

    # Check for API keys
    if not setup_api_keys():
        st.warning("Please enter all required API keys to continue.")
        return

    # Initialize components
    supabase_client = SupabaseClient()
    gemini_model = GeminiModel()
    vector_store = VectorStore(supabase_client.client, gemini_model.embeddings)

    if not supabase_client.client or not gemini_model.llm or not gemini_model.embeddings:
        return

    # Load messages
    st.session_state.messages = supabase_client.load_messages(st.session_state.session_id)

    # Initialize vector store
    vector_store.initialize(st.session_state.messages)

    # System prompt
    system_prompt = SystemMessage(content="You are a helpful assistant that can process text and images. Provide accurate and concise responses.")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
    image_path, image_url = process_image(uploaded_file)

    # Chat input
    user_input = st.text_area("Your message:", height=100)

    if st.button("Send"):
        if user_input or image_url:
            content = [{"type": "text", "text": user_input}] if user_input else []
            if image_url:
                content.append({"type": "image_url", "image_url": image_url})

            user_message = {"role": "user", "content": user_input or "Image provided", "image_path": image_path}
            st.session_state.messages.append(user_message)
            supabase_client.save_message(st.session_state.session_id, "user", user_input or "Image provided", image_path)

            messages = [system_prompt]
            context = vector_store.get_context(user_input or "Describe the image")
            if context:
                messages.append(SystemMessage(content=f"Context from conversation: {context[0].page_content}"))
            messages.append(HumanMessage(content=content))

            try:
                response = gemini_model.invoke(messages)
                assistant_message = {"role": "assistant", "content": response.content, "image_path": None}
                st.session_state.messages.append(assistant_message)
                supabase_client.save_message(st.session_state.session_id, "assistant", response.content)
                vector_store.initialize(st.session_state.messages)
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
        else:
            st.warning("Please enter a message or upload an image.")

    # Display conversation history
    st.subheader("Conversation History")
    for msg in st.session_state.messages:
        with st.container():
            if msg["role"] == "user":
                st.markdown(f"**You**: {msg['content']}")
                if msg["image_path"]:
                    st.image(msg["image_path"], width=200)
            else:
                st.markdown(f"**Assistant**: {msg['content']}")

    if st.button("Clear Chat History"):
        try:
            supabase_client.clear_history(st.session_state.session_id)
            vector_store.clear(st.session_state.session_id)
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.success("Chat history cleared!")
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")

if __name__ == "__main__":
    main()