from langchain.document_loaders import PyPDFLoader, TextLoader
import os
import tempfile
from supabase import Client
import uuid

def load_document(uploaded_file, supabase_client: Client):
    if uploaded_file is None:
        return None, None
    
    file_extension = uploaded_file.name.split('.')[-1].lower()
    document_id = str(uuid.uuid4())
    storage_path = f"documents/{document_id}.{file_extension}"
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        # Upload to Supabase Storage
        with open(tmp_file_path, "rb") as f:
            supabase_client.storage.from_("chat-documents").upload(storage_path, f, {"content-type": uploaded_file.type})
        
        # Load document content
        if file_extension == "pdf":
            loader = PyPDFLoader(tmp_file_path)
        elif file_extension == "txt":
            loader = TextLoader(tmp_file_path)
        else:
            raise ValueError("Unsupported file type. Use PDF or TXT.")
        
        documents = loader.load()
        
        # Save document metadata to Supabase
        metadata = {
            "document_id": document_id,
            "file_name": uploaded_file.name,
            "storage_path": storage_path,
            "session_id": st.session_state.session_id
        }
        supabase_client.from_("user_documents").insert(metadata).execute()
        
        return documents, document_id
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)