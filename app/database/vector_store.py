from langchain_community.vectorstores import SupabaseVectorStore

class VectorStore:
    def __init__(self, client, embeddings):
        self.client = client
        self.embeddings = embeddings
        self.vector_store = None

    def initialize(self, messages):
        try:
            texts = [msg["content"] for msg in messages if msg["role"] in ["user", "assistant"]]
            metadatas = [{"session_id": msg["session_id"], "role": msg["role"]} for msg in messages if msg["role"] in ["user", "assistant"]]
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
            self.vector_store = None
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")