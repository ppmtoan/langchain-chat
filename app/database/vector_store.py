import uuid
from langchain_community.vectorstores import SupabaseVectorStore
import streamlit as st
from typing import List

class VectorStore:
    def __init__(self, client, embeddings):
        self.client = client
        self.embeddings = embeddings
        self.vector_store = None

    def initialize(self, messages: List[dict], document_ids: List[str], session_id: str):
        try:
            texts = [msg["content"] for msg in messages if msg["role"] in ["user", "assistant"]]
            metadatas = [{"session_id": session_id, "role": msg["role"], "type": "message"} for msg in messages if msg["role"] in ["user", "assistant"]]
            
            # Fetch document content from user_documents table
            for doc_id in document_ids:
                response = self.client.from_("user_documents").select("file_name").eq("document_id", doc_id).execute()
                if response.data:
                    texts.append(response.data[0]["file_name"])  # Using file_name as a proxy; ideally, fetch document content
                    metadatas.append({"session_id": session_id, "document_id": doc_id, "type": "document"})
            
            if texts and metadatas:
                self.vector_store = SupabaseVectorStore.from_texts(
                    texts,
                    self.embeddings,
                    client=self.client,
                    table_name="documents",
                    query_name="match_documents"
                )
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")

    def add_documents(self, documents, session_id, document_id):
        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [{"session_id": session_id, "document_id": document_id, "type": "document"} for _ in documents]
            if texts and metadatas:
                self.vector_store = SupabaseVectorStore.from_texts(
                    texts,
                    self.embeddings,
                    client=self.client,
                    table_name="documents",
                    query_name="match_documents"
                )
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")

    def get_context(self, query):
        if self.vector_store:
            try:
                retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                return retriever.invoke(query)
            except Exception as e:
                print(f"Error retrieving context: {str(e)}")
        return []

    def retrieve(self, query: str) -> List:
        """Retrieve documents for LangGraph node."""
        return self.get_context(query)

    def clear(self, session_id):
        try:
            self.client.from_("documents").delete().eq("metadata->>session_id", session_id).execute()
            self.client.from_("user_documents").delete().eq("session_id", session_id).execute()
            self.vector_store = None
        except Exception as e:
            st.error(f"Error clearing vector store: {str(e)}")
            print(f"Full error details (clear): {e.__dict__}")