import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from ultralytics import YOLO
from transformers import pipeline

# Page config
st.set_page_config(page_title="WebKalorier Multi", layout="wide")
st.title("🍲 WebKalorier – Multi-Ingrediens Kalorieestimering")

# Load caloriedata
df = pd.read_csv("kaloriedata.csv")
food_list = df["navn"].tolist()

# Lazy-load models
@st.cache_resource
def load_models():
    yolo = YOLO("keremberke/yolov5m-food-detection")
    zero_shot = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32"
    )
    return yolo, zero_shot

yolo, zero_shot = load_models()

# Image uploader
uploaded = st.file_uploader("Upload et billede af din mad", type=["jpg","jpeg","png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    arr = np.array(image)
    st.image(image, caption="Originalt billede", use_column_width=True)

    # Object detection
    results = yolo(arr)
    xy = results[0].boxes.xyxy.cpu().numpy()  # [N,4]
    crops = []
    for coords in xy:
        x1, y1, x2, y2 = coords.astype(int)
        crop = image.crop((x1, y1, x2, y2))
        crops.append(crop)

    st.subheader("Detekterede ingrediensbilleder")
    final_labels = []
    for i, crop in enumerate(crops):
        st.image(crop, width=150, caption=f"Ingrediens {i+1}")
        # Zero-shot classification
        res = zero_shot(crop, candidate_labels=food_list)
        label = res["labels"][0]
        score = res["scores"][0]
        st.write(f"Gæt: **{label}** ({score*100:.1f}% sikkerhed)")
        if score < 0.7:
            choice = st.selectbox(
                f"Usikker - vælg ingrediens {i+1}",
                options=food_list,
                key=f"fb_{i}"
            )
        else:
            choice = label
        final_labels.append(choice)

    # Quantity input
    st.subheader("Angiv mængde pr. ingrediens (gram)")
    quantities = {}
    for i, label in enumerate(final_labels):
        quantities[label] = st.number_input(
            f"{label} (g):", min_value=1, max_value=2000, value=100, key=f"qty_{i}"
        )

    # Compute and display calories
    st.subheader("📊 Kalorieanalyse")
    total = 0.0
    for label, gram in quantities.items():
        kcal100 = float(df.loc[df["navn"] == label, "kcal_pr_100g"])
        kcal = gram * kcal100 / 100
        total += kcal
        st.write(f"- {gram} g **{label}** → {kcal:.1f} kcal")
    st.markdown(f"**Total: {total:.1f} kcal**")

    # Feedback log
    feedback = st.text_input("Feedback eller rettelse af samlet resultat (valgfrit)")
    if st.button("Send feedback"):
        with open("feedback_log.csv", "a") as f:
            f.write(",".join(final_labels) + f",Total,{total:.1f}," + feedback + "\n")
        st.success("Tak for din feedback!")