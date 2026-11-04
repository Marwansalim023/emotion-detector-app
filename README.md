# 🎭 AI Emotion Detector

An end-to-end deep learning project that detects **6 human emotions** in English text using **transfer learning**.

**Live app:** https://ai-emotion-detector-mehrwan.streamlit.app

| 😢 Sadness | 😄 Joy | ❤️ Love | 😠 Anger | 😨 Fear | 😲 Surprise |
|---|---|---|---|---|---|

## Problem Statement

Can a computer understand human feelings from text? This project fine-tunes a pretrained Transformer model to read a sentence and classify the emotion behind it. Emotion detection can help businesses sort customer feedback, help support teams spot angry messages first, and track public mood on social media.

## Dataset

- **Name:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) (Hugging Face)
- **Size:** 20,000 English Twitter messages
- **Split:** 16,000 train / 2,000 validation / 2,000 test
- **Classes:** sadness, joy, love, anger, fear, surprise
- **Preprocessing:** exploratory analysis (class distribution, sentence length), removed duplicate and empty texts, tokenized with the DistilBERT tokenizer (max 128 tokens), dynamic padding

## Model

- **Pretrained model:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased), a compressed version of Google's BERT (~66M parameters)
- **Method:** transfer learning. A new 6-class classification head was added and the whole model was fine-tuned on the emotion dataset.
- **Training:** 2 epochs, batch size 32, learning rate 5e-5, Google Colab T4 GPU
- **Test accuracy:** 93.0% on 2,000 unseen test sentences
- **Fine-tuned model on Hugging Face:** [MarwanSalim023/emotion-detector](https://huggingface.co/MarwanSalim023/emotion-detector)

## How It Works

```
Pretrained DistilBERT (Hugging Face)
        ↓  fine-tune in Google Colab
Fine-tuned model → uploaded to Hugging Face
        ↓  loaded by app.py
GitHub repository → connected to Streamlit Community Cloud
        ↓
Live web app 🎉
```

## Repository Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit web app: loads the model from Hugging Face and shows predictions |
| `requirements.txt` | Python libraries Streamlit installs (CPU version of PyTorch) |
| `emotion_detector.ipynb` | Colab notebook: EDA, cleaning, fine-tuning, evaluation and upload |
| `README.md` | Project documentation |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use the Model in Python

```python
from transformers import pipeline

clf = pipeline("text-classification", model="MarwanSalim023/emotion-detector")
print(clf("I just got my first job offer!"))
```

## Author

Marwan Salim · BS Information Technology · University of Gwadar
