import torch
import open_clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model.to(device)

def extract_clip_features(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image).cpu().numpy().flatten()
    return features
