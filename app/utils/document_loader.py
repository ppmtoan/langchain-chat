import streamlit as st
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
    
    # Save uploaded file temporarily for processing
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Upload to Supabase Storage
        try:
            with open(tmp_file_path, "rb") as f:
                supabase_client.storage.from_("chat-documents").upload(
                    storage_path, 
                    f, 
                    {"content-type": uploaded_file.type}
                )
        except Exception as e:
            st.error(f"Error uploading document to Supabase Storage: {str(e)}")
            return None, None
        
        # Load document content
        try:
            if file_extension == "pdf":
                loader = PyPDFLoader(tmp_file_path)
            elif file_extension == "txt":
                loader = TextLoader(tmp_file_path)
            else:
                st.error("Unsupported file type. Use PDF or TXT.")
                return None, None
            
            documents = loader.load()
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return None, None
        
        # Save document metadata to Supabase
        try:
            metadata = {
                "document_id": document_id,
                "file_name": uploaded_file.name,
                "storage_path": storage_path,
                "session_id": st.session_state.session_id,
                "public_url": supabase_client.storage.from_("chat-documents").get_public_url(storage_path)
            }
            supabase_client.from_("user_documents").insert(metadata).execute()
        except Exception as e:
            st.error(f"Error saving document metadata: {str(e)}")
            return None, None
        
        return documents, document_id
    
    except Exception as e:
        st.error(f"Error handling temporary file: {str(e)}")
        return None, None
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                st.warning(f"Error cleaning up temporary file: {str(e)}")