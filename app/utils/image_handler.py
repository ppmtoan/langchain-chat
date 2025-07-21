import base64
import os
import uuid

def process_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_path = f"images/{uuid.uuid4()}.png"
        os.makedirs("images", exist_ok=True)
        with open(image_path, "wb") as f:
            f.write(bytes_data)
        encoded_image = base64.b64encode(bytes_data).decode("utf-8")
        return image_path, f"data:image/png;base64,{encoded_image}"
    return None, None