import streamlit as st
from PIL import Image
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="WebKalorier", layout="centered")
st.title("🍽️ WebKalorier – Kalorieestimering")

# Load calorie data
df = pd.read_csv("kaloriedata.csv")
food_list = df["navn"].tolist()

@st.cache_resource
def get_classifier():
    return pipeline("image-classification", model="eslamxm/vit-base-food101")

uploaded_file = st.file_uploader("Upload et billede af din mad", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploadet billede", use_column_width=True)

    with st.spinner("Analyserer..."):
        classifier = get_classifier()
        results = classifier(image, top_k=3)

    label = results[0]["label"].replace("_", " ").lower()
    score = results[0]["score"]
    st.markdown(f"**Modelgæt:** {label} ({score*100:.1f}% sikkerhed)")

    if score < 0.7:
        chosen = st.selectbox("Usikker – vælg manuelt:", food_list)
    else:
        chosen = label

    gram = st.number_input(f"Hvor mange gram {chosen}?", 1, 2000, 100)
    kcal = float(df.loc[df["navn"] == chosen, "kcal_pr_100g"]) * gram / 100
    st.markdown(f"### Analyse: {gram} g {chosen} = {kcal:.1f} kcal")

    feedback = st.text_input("Feedback (valgfrit)")
    if st.button("Send feedback"):
        with open("feedback_log.csv", "a") as f:
            f.write(f"{chosen},{score:.2f},{feedback}\n")
        st.success("Tak for din feedback!")