🧠 Mental Health Text Classification using BERT
This project focuses on classifying user-generated text into mental health categories using Transformer-based models, specifically BERT, in a Kaggle-compatible environment. It explores preprocessing techniques, handling class imbalance, and model fine-tuning using Hugging Face Transformers.

🎯 Objective
To build a text classification model that can accurately detect different mental health conditions (e.g., anxiety, depression) from social media or forum text using state-of-the-art NLP techniques.

⚙️ Tech Stack
Language: Python

Frameworks & Libraries:

pandas, numpy – Data handling

matplotlib, seaborn – Visualization

scikit-learn – Evaluation & preprocessing

transformers (Hugging Face) – BERT model and tokenizer

nltk – NLP preprocessing (stopwords)

torch – PyTorch backend for model training

imblearn – Oversampling for class imbalance

📁 Dataset
Source: Combined Data.csv (loaded from Kaggle dataset)

Description: Text samples labeled with corresponding mental health conditions.

🧼 Preprocessing
Removal of stopwords using nltk

Regex cleaning

Label encoding

Oversampling using RandomOverSampler to balance classes

🧠 Model
Base model: bert-base-uncased (from Hugging Face Transformers)

Fine-tuned using Trainer and TrainingArguments classes

Tokenized using BertTokenizer

📊 Evaluation
Classification report (Precision, Recall, F1-score)

Confusion Matrix
📌 Highlights
Uses RandomOverSampler to tackle imbalanced data

Completely offline mode for W&B logging

Utilizes Hugging Face’s easy-to-use Trainer API

Optimized for GPU runtime on Kaggle (e.g., Tesla T4)

🚀 Future Improvements
Experiment with larger models like roberta-base, distilbert

Deploy via REST API using FastAPI or Flask

UI for user to input custom text and get classification
