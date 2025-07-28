import base64
import uuid
from supabase import Client

def process_image(uploaded_file, supabase_client: Client):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_id = str(uuid.uuid4())
        image_path = f"images/{image_id}.png"
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")
        
        # Upload to Supabase Storage
        try:
            supabase_client.storage.from_("chat-images").upload(image_path, bytes_data, {"content-type": uploaded_file.type})
            # Generate public URL for the image
            image_url = supabase_client.storage.from_("chat-images").get_public_url(image_path)
            return image_path, image_url
        except Exception as e:
            print(f"Error uploading image to Supabase: {str(e)}")
            return None, None
    return None, None