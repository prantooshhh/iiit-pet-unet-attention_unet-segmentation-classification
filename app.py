import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Import model architectures
from model_architecture import UNet, AttentionUNet

# Class names mapping (0-36 IDs)
BREED_CLASSES = [
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound",
    "beagle", "bengal", "birman", "bombay", "boxer", "brabancon_griffon",
    "british_shorthair", "chihuahua", "egyptian_mau", "english_cocker_spaniel",
    "english_setter", "german_shorthaired", "great_dane", "havanese",
    "japanese_chin", "keeshond", "maine_coon", "miniature_pinscher",
    "newfoundland", "newspaper_cat", "pomeranian", "pug", "ragdoll",
    "russian_blue", "saint_bernard", "samoyed", "scottish_terrier",
    "shiba_inu", "siamese", "sphynx", "staffordshire_bull_terrier",
    "wheaten_terrier", "yorkshire_terrier"
]

st.set_page_config(page_title="Pet Segmentation & Classification", layout="wide")
st.title("Pet Image Segmentation & Breed Classification")

# Load model weights from HF Hub
@st.cache_resource
def get_model(model_choice):
    if model_choice == "Standard UNet":
        filename = "unet_model.safetensors"
        model = UNet()
    else:
        filename = "attention_unet_model.safetensors"
        model = AttentionUNet()

    # Download from Hugging Face Model repo
    # REPLACE 'YOUR_HF_USERNAME' with your actual username
    weights_path = hf_hub_download(
        repo_id="prantooshhh/pet-segmentation-models", 
        filename=filename
    )
    
    state_dict = load_file(weights_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Transforms to match notebook training pipeline
img_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a Pet Image", type=["jpg", "jpeg", "png"])
    model_choice = st.selectbox("Select Model", ["Standard UNet", "Attention UNet"])
    analyze_btn = st.button("Run Inference")

if analyze_btn and uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")
    
    # Preprocessing
    input_tensor = img_transform(raw_img).unsqueeze(0)
    
    # Inference
    model = get_model(model_choice)
    with torch.no_grad():
        mask_logits, class_logits = model(input_tensor)
    
    # Process classification result
    pred_class_id = torch.argmax(class_logits, dim=1).item()
    pred_breed = BREED_CLASSES[pred_class_id] if pred_class_id < len(BREED_CLASSES) else "Unknown"
    
    # Process segmentation mask
    mask = torch.sigmoid(mask_logits).squeeze().numpy() > 0.5
    
    # Overlay mask on resized image
    resized_raw = raw_img.resize((256, 256))
    img_np = np.array(resized_raw)
    
    overlay = img_np.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8) # Red overlay
    
    with col2:
        st.subheader("Results")
        st.image(overlay, caption="Segmentation Overlay (Red = Pet)", use_column_width=True)
        st.success(f"**Predicted Breed:** {pred_breed}")