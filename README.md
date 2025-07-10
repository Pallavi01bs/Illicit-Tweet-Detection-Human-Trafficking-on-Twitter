# 🚨 Illicit Tweet Detection: Human Trafficking on Twitter
A machine learning-based classifier using **SVM (Support Vector Machine)** and **NLP techniques** to detect tweets suspected of promoting human trafficking or sexual exploitation, especially involving minors.

## 🔍 Project Overview
This project aims to build a classification model that can identify suspicious or illicit tweets linked to human trafficking. Using **NLP preprocessing**, **TF-IDF vectorization**, and **SVM**, the model learns to classify tweets as either **positive (suspicious)** or **negative (non-suspicious)**. It is particularly useful for supporting organizations that monitor trafficking activity online.

## 🛠️ Tech Stack
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- TF-IDF Vectorizer
- Pickle
- Jupyter Notebook

## 📁 Dataset
The dataset is based on the **Sentiment140 Twitter Dataset**, which contains  **1.6 million tweets**. Important columns include:

- text: Tweet content
- target: Label (positive/negative)
- user, id, date, flag: Additional metadata

    📌 Note: If you're running locally, replace any cloud-specific paths with your own (e.g., ./data/tweets.csv).

## 📊 Key Features
  Text Preprocessing:
   - Lowercasing, removing hyperlinks, punctuations, and stopwords
   - Tokenization, stemming, and lemmatization

 Feature Engineering:
   - TF-IDF vectorization to convert text to numerical format

 Model Building:
   - Trained using Support Vector Machine (SVM)
   - Serialized with Pickle for reuse

 Evaluation Metrics:
   - Accuracy Score
   - Confusion Matrix
   - ROC-AUC Curve

 Real-time Testing:
   - Load saved model
   - Predict new tweets (Positive / Negative)
