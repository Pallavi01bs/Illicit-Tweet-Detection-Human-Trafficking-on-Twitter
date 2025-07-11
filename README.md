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

## 🚀 How to Run the Project
1. Clone the Repository
```bash
git clone https://github.com/pallavi01bs/IllicitTweetDetection.git
cd IllicitTweetDetection
```

2. Install Dependencies
```bash
pip install -r requirements.txt
#or manually:
pip install numpy pandas matplotlib seaborn nltk scikit-learn wordcloud
```

3. Run the Notebooks
 - PT1 training.ipynb — Train the SVM model on preprocessed data
 - PT1 testing.ipynb — Load the model and classify new tweets

## 🧪 Sample Output
|Tweet	|Prediction|
|-----|------|
|"Just like #EpsteinClientList"	|Positive|
|"https://t.co/IMWpn3S1E4"	|Negative|

## ✅ System Requirements
Hardware
 - 1.1 GHz+ Dual Core Processor
 - 8 GB RAM minimum

Software
 - Windows 11
 - Python 3.7 or higher
 - Jupyter Notebook / Google Colab

## 📚 References
 - Sentiment140 Dataset
 - Scikit-learn SVM Docs
 - NLTK
- Research Papers (see Illicit message...docx for full list)

  ## 📄 License
This project is licensed under the [MIT License](LICENSE).
