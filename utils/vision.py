import base64

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"
