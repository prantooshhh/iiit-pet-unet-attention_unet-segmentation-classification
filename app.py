import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Import model architectures from model_architectures.py
from model_architectures import UNet, AttentionUNet

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
        model = UNet(n_channels=3, n_seg_classes=1, n_breed_classes=37)
    else:
        filename = "att_unet_model.safetensors"  # Matches tester_att_unet.py
        model = AttentionUNet(n_channels=3, n_seg_classes=1, n_breed_classes=37)

    # REMINDER: Ensure 'YOUR_HF_USERNAME' matches your Hugging Face username
    weights_path = hf_hub_download(
        repo_id="prantooshhh/pet-segmentation-models", 
        filename=filename
    )
    
    safetensor_weights = load_file(weights_path)
    model.load_state_dict(safetensor_weights, strict=True)
    model.eval()
    return model

# Preprocessing matching tester scripts
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload a Pet Image", type=["jpg", "jpeg", "png"])
    model_choice = st.selectbox("Select Model", ["Standard UNet", "Attention UNet"])
    analyze_btn = st.button("Run Inference")

if analyze_btn and uploaded_file is not None:
    orig_img = Image.open(uploaded_file).convert("RGB")
    orig_width, orig_height = orig_img.size
    
    # Preprocessing & batch expansion
    input_tensor = preprocess(orig_img).unsqueeze(0)
    
    # Inference execution
    model = get_model(model_choice)
    with torch.no_grad():
        seg_logits, breed_logits = model(input_tensor)
    
    # Remove batch dimension
    seg_logits = seg_logits.squeeze(0)
    breed_logits = breed_logits.squeeze(0)
    
    # Process classification
    breed_probs = F.softmax(breed_logits, dim=0)
    pred_class_id = torch.argmax(breed_probs).item()
    confidence = breed_probs[pred_class_id].item() * 100
    pred_breed = BREED_CLASSES[pred_class_id] if pred_class_id < len(BREED_CLASSES) else "Unknown"
    
    # Process segmentation (matching tester interpolation logic)
    probabilities = torch.sigmoid(seg_logits)
    probabilities = probabilities.unsqueeze(0)
    probabilities = F.interpolate(probabilities, size=(orig_height, orig_width), mode="bilinear", align_corners=False)
    probabilities = probabilities.squeeze(0).squeeze(0)
    binary_mask = (probabilities > 0.5).byte().numpy()
    
    # Generate translucent red overlay
    mask_rgba = Image.new("RGBA", (orig_width, orig_height), (0, 0, 0, 0))
    mask_pixels = mask_rgba.load()
    for y in range(orig_height):
        for x in range(orig_width):
            if binary_mask[y, x] == 1:
                mask_pixels[x, y] = (255, 0, 0, 115)  # Red with ~45% opacity
                
    orig_rgba = orig_img.convert("RGBA")
    blended_image = Image.alpha_composite(orig_rgba, mask_rgba).convert("RGB")
    
    with col2:
        st.subheader("Results")
        st.image(blended_image, caption="Segmentation Overlay (Red = Pet)", use_container_width=True)
        st.success(f"**Predicted Breed:** {pred_breed} ({confidence:.2f}% confidence)")
