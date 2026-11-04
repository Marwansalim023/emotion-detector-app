import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="AI Emotion Detector", page_icon="🎭")

EMOJI = {"joy": "😄", "sadness": "😢", "love": "❤️",
         "anger": "😠", "fear": "😨", "surprise": "😲"}

@st.cache_resource
def load_model():
    return pipeline("text-classification",
                    model="MarwanSalim023/emotion-detector", top_k=None)

clf = load_model()

st.title("🎭 AI Emotion Detector")
st.write("Type any sentence and the AI will detect the emotion behind it.")

examples = ["(type your own)",
            "I just got my first job offer!",
            "Nobody came to my birthday party.",
            "How dare you lie to me!",
            "I heard a strange noise outside my window at night."]
choice = st.selectbox("Try an example:", examples)
text = st.text_area("Your sentence:", value="" if choice == examples[0] else choice)

if st.button("Detect emotion"):
    if text.strip():
        scores = sorted(clf([text])[0], key=lambda s: s["score"], reverse=True)
        top = scores[0]
        st.markdown(f"## {EMOJI[top['label']]} {top['label'].capitalize()}")
        st.write(f"Confidence: **{top['score']:.1%}**")

        df = pd.DataFrame({"Emotion": [s["label"] for s in scores],
                           "Score": [s["score"] for s in scores]}).set_index("Emotion")
        st.bar_chart(df)
    else:
        st.warning("Please type a sentence first.")
