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
from langchain.prompts import PromptTemplate
from langsmith import Client as LangSmithClient
from langgraph.graph import StateGraph, START, END
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

# Add current working directory to Python path
sys.path.append(os.getcwd())

# Define fallback RAG prompt
FALLBACK_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="You are an assistant for question-answering tasks. Use the following context to answer the question. If you don't know the answer, say so. If no context is provided, answer based on your knowledge. Keep answers concise.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
)

# Define LangGraph state
class RAGState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    image_url: str

# Define LangGraph nodes
def retrieve(state: RAGState) -> RAGState:
    query = state["question"] or "Describe the image"
    context = vector_store.retrieve(query)
    print(f"Retrieved {len(context)} documents for query: {query}")
    return {"question": state["question"], "context": context, "answer": state["answer"], "image_url": state["image_url"]}

def generate(state: RAGState) -> RAGState:
    try:
        # Initialize LangSmith client
        langsmith_client = LangSmithClient()
        # Pull prompt from LangSmith (replacing langchain.hub)
        prompt = langsmith_client.pull_prompt("rlm/rag-prompt")
    except Exception as e:
        st.warning(f"Failed to load rlm/rag-prompt from LangSmith: {str(e)}. Using fallback prompt. Ensure LANGSMITH_API_KEY is set.")
        print(f"Error loading prompt: {str(e)}")
        prompt = FALLBACK_RAG_PROMPT
    context_text = "\n".join([doc.page_content for doc in state["context"]]) if state["context"] else "No relevant context found."
    content = [{"type": "text", "text": prompt.format(context=context_text, question=state["question"] or "Describe the image")}]
    if state["image_url"]:
        content.append({"type": "image_url", "image_url": state["image_url"]})
    messages = [
        SystemMessage(content="You are a helpful assistant that can process text, images, and documents. Provide accurate and concise responses, using provided document context when relevant."),
        HumanMessage(content=content)
    ]
    try:
        response = gemini_model.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        st.error(f"Error invoking Gemini model: {str(e)}")
        print(f"Full error details (generate): {e.__dict__}")
        return state
    print(f"Generated answer: {answer[:100]}...")
    return {"question": state["question"], "context": state["context"], "answer": answer, "image_url": state["image_url"]}

# Build LangGraph workflow
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
rag_chain = workflow.compile()

# Initialize components (global for LangGraph)
document_loader, supabase_client, gemini_model, vector_store = None, None, None, None

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "document_ids" not in st.session_state:
        st.session_state.document_ids = []
    if "vector_store_initialized" not in st.session_state:
        st.session_state.vector_store_initialized = False

def load_css():
    """Load custom CSS styles."""
    with open("app/templates/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def setup_api_keys():
    """Configure API keys from secrets or user input."""
    if "SUPABASE_URL" not in st.secrets or "SUPABASE_KEY" not in st.secrets or "GOOGLE_API_KEY" not in st.secrets or "LANGSMITH_API_KEY" not in st.secrets:
        st.subheader("Enter API Keys")
        supabase_url = st.text_input("Supabase URL:", type="default")
        supabase_key = st.text_input("Supabase Service Role Key:", type="password")
        google_api_key = st.text_input("Google AI API Key:", type="password")
        langsmith_api_key = st.text_input("LangSmith API Key:", type="password")
        if supabase_url and supabase_key and google_api_key and langsmith_api_key:
            st.session_state["SUPABASE_URL"] = supabase_url
            st.session_state["SUPABASE_KEY"] = supabase_key
            st.session_state["GOOGLE_API_KEY"] = google_api_key
            st.session_state["LANGSMITH_API_KEY"] = langsmith_api_key
            st.success("API keys set successfully!")
            return False
    else:
        os.environ[Config.SUPABASE_URL] = st.secrets["SUPABASE_URL"]
        os.environ[Config.SUPABASE_KEY] = st.secrets["SUPABASE_KEY"]
        os.environ[Config.GOOGLE_API_KEY] = st.secrets["GOOGLE_API_KEY"]
        os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
        return True
    return False

def initialize_components():
    """Initialize core components."""
    global document_loader, supabase_client, gemini_model, vector_store
    document_loader = DocumentLoader()
    supabase_client = SupabaseClient()
    gemini_model = GeminiModel()
    vector_store = VectorStore(supabase_client.client, gemini_model.embeddings)
    return document_loader, supabase_client, gemini_model, vector_store

def handle_document_upload(document_loader, supabase_client, vector_store):
    """Handle document upload and processing."""
    with st.sidebar:
        st.subheader("Upload Document")
        uploaded_doc = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"], key="document_uploader")
        if uploaded_doc:
            try:
                documents, document_id = document_loader.load_document(uploaded_doc, supabase_client.client, st.session_state.session_id)
                if documents:
                    vector_store.add_documents(documents, st.session_state.session_id, document_id)
                    st.session_state.document_ids.append(document_id)
                    st.session_state.vector_store_initialized = False  # Force reinitialization
                    st.success(f"Document {uploaded_doc.name} loaded successfully!")
            except Exception as e:
                st.error(f"Error loading document: {str(e)}")
                print(f"Full error details (upload_document): {e.__dict__}")

def handle_image_upload(supabase_client):
    """Handle image upload and processing."""
    with st.sidebar:
        st.subheader("Upload Image")
        uploaded_image = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"], key="image_uploader")
        return process_image(uploaded_image, supabase_client.client) if uploaded_image else (None, None)

def process_user_input(user_input, image_url, supabase_client, vector_store):
    """Process user input and generate response using LangGraph."""
    if user_input or image_url:
        user_message = {"role": "user", "content": user_input or "Image provided", "image_url": image_url}
        st.session_state.messages.append(user_message)
        supabase_client.save_message(st.session_state.session_id, "user", user_input or "Image provided", image_url)

        try:
            # Run LangGraph RAG pipeline
            result = rag_chain.invoke({
                "question": user_input,
                "context": [],
                "answer": "",
                "image_url": image_url
            })
            answer = result["answer"]
            if not answer:
                st.error("No response generated. Check LangSmith or Gemini model configuration.")
                return
            assistant_message = {"role": "assistant", "content": answer, "image_url": None}
            st.session_state.messages.append(assistant_message)
            supabase_client.save_message(st.session_state.session_id, "assistant", answer)
            vector_store.initialize(st.session_state.messages, st.session_state.document_ids, st.session_state.session_id)
            st.session_state.vector_store_initialized = True
        except Exception as e:
            st.error(f"Error processing input: {str(e)}")
            print(f"Full error details (process_user_input): {e.__dict__}")
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
            st.session_state.vector_store_initialized = False
            st.success("Chat history and documents cleared!")
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")
            print(f"Full error details (clear_history): {e.__dict__}")

def main():
    """Main application function."""
    st.set_page_config(page_title="Gemini Chat App with Supabase", layout="wide")
    st.title("Gemini Chat App with Supabase")
    
    initialize_session_state()
    load_css()

    if not setup_api_keys():
        st.warning("Please enter all required API keys (including LANGSMITH_API_KEY) to continue.")
        return

    global document_loader, supabase_client, gemini_model, vector_store
    document_loader, supabase_client, gemini_model, vector_store = initialize_components()
    
    if not supabase_client.client or not gemini_model.llm or not gemini_model.embeddings:
        st.error("Failed to initialize components. Check API keys and configuration.")
        return

    if not st.session_state.vector_store_initialized:
        try:
            vector_store.initialize(st.session_state.messages, st.session_state.document_ids, st.session_state.session_id)
            st.session_state.vector_store_initialized = True
        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            print(f"Full error details (initialize): {e.__dict__}")
            st.stop()

    st.session_state.messages = supabase_client.load_messages(st.session_state.session_id)
    
    handle_document_upload(document_loader, supabase_client, vector_store)
    image_path, image_url = handle_image_upload(supabase_client)
    
    user_input = st.text_area("Your message:", height=100)
    
    if st.button("Send"):
        process_user_input(user_input, image_url, supabase_client, vector_store)
    
    display_conversation_history()
    clear_chat_history(supabase_client, vector_store)

if __name__ == "__main__":
    main()