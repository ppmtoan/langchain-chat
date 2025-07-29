import uuid
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from supabase import Client
import os

class DocumentLoader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def load_document(self, file,  supabase_client: Client, session_id):
        try:
            document_id = str(uuid.uuid4())
            storage_path = f"documents/{document_id}.{file.name.split('.')[-1]}"
            
            # Upload to Supabase storage
            file_content = file.getvalue()
            response = supabase_client.storage.from_("chat-documents").upload(storage_path, file=file_content)
            
            if response.status_code in (200, 201):
                public_url = supabase_client.storage.from_("chat-documents").get_public_url(storage_path)
                supabase_client.from_("user_documents").insert({
                    "document_id": document_id,
                    "file_name": file.name,
                    "storage_path": storage_path,
                    "session_id": session_id,
                    "public_url":  public_url
                }).execute()
                
                # Process PDF or text file
                documents = []
                if file.name.endswith(".pdf"):
                    try:
                        # Save file temporarily for PyPDFLoader
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                            temp_file.write(file_content)
                            temp_file_path = temp_file.name
                        
                        # Load PDF using PyPDFLoader
                        loader = PyPDFLoader(temp_file_path)
                        pdf_documents = loader.load()
                        
                        # Clean up temporary file
                        os.unlink(temp_file_path)
                        
                        if not pdf_documents:
                            st.error("No text extracted from PDF. The file may be scanned or empty.")
                            return None, None
                        
                        # Split documents and add metadata
                        texts = [doc.page_content for doc in pdf_documents]
                        metadatas = [{"document_id": document_id} for _ in pdf_documents]
                        documents = self.text_splitter.create_documents(texts, metadatas=metadatas)
                        print(f"Extracted {len(texts)} pages from PDF, split into {len(documents)} chunks")
                    except Exception as e:
                        st.error(f"Error processing PDF: {str(e)}")
                        print(f"Full error details (PDF processing): {e.__dict__}")
                        return None, None
                elif file.name.endswith(".txt"):
                    try:
                        text = file_content.decode("utf-8")
                        documents = self.text_splitter.create_documents([text], metadatas=[{"document_id": document_id}])
                        print(f"Extracted text length: {len(text)} characters, split into {len(documents)} chunks")
                    except Exception as e:
                        st.error(f"Error processing TXT: {str(e)}")
                        print(f"Full error details (TXT processing): {e.__dict__}")
                        return None, None
                else:
                    st.error("Unsupported file type. Please upload a PDF or TXT file.")
                    return None, None
                
                return documents, document_id
            else:
                st.error(f"Failed to upload file to storage: {response.status_code}")
                print(f"Storage upload response: {response}")
                return None, None
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
            print(f"Full error details (load_document): {e.__dict__}")
            return None, None