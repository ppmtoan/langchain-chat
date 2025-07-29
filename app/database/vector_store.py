from langchain_community.vectorstores import SupabaseVectorStore

class VectorStore:
    def __init__(self, client, embeddings):
        self.client = client
        self.embeddings = embeddings
        self.vector_store = None

    def initialize(self, messages, document_ids, session_id):
        try:
            texts = [msg["content"] for msg in messages if msg["role"] in ["user", "assistant"]]
            metadatas = [{"session_id": msg["session_id"], "role": msg["role"], "type": "message"} for msg in messages if msg["role"] in ["user", "assistant"]]
            
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
                return self.vector_store.as_retriever().invoke(query)
            except Exception as e:
                print(f"Error retrieving context: {str(e)}")
        return []

    def clear(self, session_id):
        try:
            self.client.from_("documents").delete().eq("metadata->>session_id", session_id).execute()
            self.client.from_("user_documents").delete().eq("session_id", session_id).execute()
            self.vector_store = None
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")