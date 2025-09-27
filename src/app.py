import streamlit as st
import torch
import timm
from torchvision import transforms, datasets
from PIL import Image
from pathlib import Path

# --- Paths ---
ROOT = Path("data/raw")
MODEL_PATH = Path("models/best_model.pth")


classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]


# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
@st.cache_resource  # cache model so it loads only once
def load_model():
    model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=len(classes))
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

# --- Transform for inference ---
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Care tips dictionary (shortened examples, you can expand) ---
care_tips = {
    "Tomato___Early_blight": "Remove infected leaves, avoid overhead watering, and rotate crops.",
    "Tomato___Late_blight": "Destroy infected plants, use fungicide sprays, ensure proper spacing.",
    "Potato___Early_blight": "Remove affected foliage, apply fungicide, improve soil drainage.",
    "Apple___Apple_scab": "Prune trees for airflow, remove infected leaves, apply fungicide.",
    "Apple___Cedar_apple_rust": "Remove nearby junipers, prune affected branches, use resistant varieties.",
    "healthy": "Your plant looks healthy! Keep monitoring it with good watering and sunlight."
}

# --- Streamlit UI ---
st.set_page_config(page_title="Plant Doctor", page_icon="🌱", layout="centered")

st.title("🌱 Plant Doctor")
st.subheader("Upload a leaf photo to detect plant disease with AI")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image",  use_container_width=True)

    # --- Prediction ---
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top3 = torch.topk(probs, 3)

    # --- Results ---
    st.markdown("### 🩺 Diagnosis Result")
    main_class = classes[top3.indices[0]]
    main_conf = top3.values[0].item() * 100

    st.success(f"**Prediction:** {main_class} ({main_conf:.2f}%)")

    # Confidence bar
    st.progress(int(main_conf))

    # # Top-3 predictions
    # st.markdown("#### 🔎 Top-3 Predictions")
    # for i in range(3):
    #     cls = classes[top3.indices[i]]
    #     conf = top3.values[i].item() * 100
    #     st.write(f"{cls} — {conf:.2f}%")
    #     st.progress(int(conf))

    # Care tip
    st.markdown("#### 🌿 Care Tip")
    tip = care_tips.get(main_class, "General advice: Remove affected leaves, maintain good watering and use organic treatment if needed.")
    st.info(tip)

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.markdown("""
**Plant Doctor** uses an AI model (MobileNetV3) trained on 87,000 images of plant leaves  
to detect 38 different plant diseases.  
Upload a clear image of a single leaf for the best results.
""")
st.sidebar.markdown("[GitHub Repo](https://github.com/zeinahassoun93/plant-doctor)")
