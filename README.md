# 🧠 Mental Health Text Classification using BERT

This project aims to classify user-generated text into different mental health conditions using transformer-based models, specifically BERT. The model is fine-tuned on labeled text data and helps in early detection of mental health issues through Natural Language Processing (NLP).

## 🎯 Objective

To develop an intelligent system that can detect mental health issues like anxiety, depression, etc., from written text using machine learning—particularly BERT-based deep learning models.

## ⚙️ Tech Stack

- **Programming Language:** Python
- **Libraries & Frameworks:**
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Visualization
  - `nltk`, `re` – Text preprocessing
  - `scikit-learn` – Evaluation metrics, preprocessing
  - `transformers (Hugging Face)` – BERT model/tokenizer, training API
  - `torch` – Backend for training
  - `imblearn` – Random over-sampling to handle class imbalance
  - `joblib` – Saving models (optional)

## 📁 Dataset

- **Source:** Kaggle Dataset
- **File:** `Combined Data.csv`
- **Columns:** Text, Label (mental health issue category)

The dataset contains posts or messages labeled with various mental health conditions.

## 🧹 Preprocessing Steps

- Removal of stopwords using `nltk`
- Lowercasing and regex cleaning
- Label encoding for categories
- Oversampling using `RandomOverSampler` to balance classes

## 🧠 Model Details

- **Model:** `bert-base-uncased`
- **Tokenizer:** `BertTokenizer` from Hugging Face
- **Training:** Done using Hugging Face `Trainer` API
- **Training Arguments:**
  - Batch size
  - Epochs
  - Logging/Offline mode for W&B
- **Platform:** Compatible with GPU (e.g., Kaggle with NVIDIA Tesla T4)

## 📊 Evaluation

Each model is evaluated using:

- Confusion Matrix
- Classification Report
- Accuracy, Precision, Recall, F1-score

