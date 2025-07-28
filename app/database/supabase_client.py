from supabase import create_client, Client
import os
from app.config import Config
from datetime import datetime
import streamlit as st

class SupabaseClient:
    def __init__(self):
        try:
            self.client = create_client(os.environ[Config.SUPABASE_URL], os.environ[Config.SUPABASE_KEY])
        except Exception as e:
            st.error(f"Error initializing Supabase client: {str(e)}")
            self.client = None

    def load_messages(self, session_id):
        try:
            response = self.client.from_("messages").select("*").eq("session_id", session_id).order("timestamp").execute()
            return [
                {"role": msg["role"], "content": msg["content"], "image_url": msg["image_url"]}
                for msg in response.data
            ]
        except Exception as e:
            st.error(f"Error loading messages: {str(e)}")
            return []

    def save_message(self, session_id, role, content, image_url=None):
        try:
            data = {
                "session_id": session_id,
                "role": role,
                "content": content,
                "image_url": image_url,
                "timestamp": datetime.utcnow().isoformat()
            }
            self.client.from_("messages").insert(data).execute()
        except Exception as e:
            st.error(f"Error saving message: {str(e)}")

    def clear_history(self, session_id):
        try:
            self.client.from_("messages").delete().eq("session_id", session_id).execute()
        except Exception as e:
            st.error(f"Error clearing history: {str(e)}")