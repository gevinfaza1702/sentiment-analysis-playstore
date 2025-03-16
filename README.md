# 📊 Sentiment Analysis on Google Play Store Reviews

## 📌 Project Overview
This project focuses on **Sentiment Analysis of Google Play Store Reviews** using **Natural Language Processing (NLP) and Machine Learning (ML)**. It classifies user reviews into **positive, neutral, or negative** sentiments to gain insights into user experiences.

## 🚀 Features
- **Scraping Data**: Extracting user reviews from Google Play Store using `google-play-scraper`.
- **Preprocessing**: Cleaning the text by removing special characters, duplicates, and empty reviews.
- **Sentiment Labeling**: Using **Lexicon-Based Sentiment Analysis (VADER)** to classify sentiment based on text content.
- **Feature Extraction**: Using **TF-IDF Vectorization** to convert text into numerical form.
- **Model Training**: Implementing **Support Vector Machine (SVM)** to classify sentiment.
- **Evaluation**: Measuring model accuracy and fine-tuning hyperparameters.
- **Inference**: Predicting sentiment on new reviews.

## 📂 Project Structure
📂 sentiment-analysis-playstore/ │── 📄 analisis_sentimen.ipynb # Jupyter Notebook │── 📄 scraping_playstore.py # Play Store data scraper │── 📄 preprocessing.py # Text preprocessing │── 📄 labeling.py # Sentiment labeling │── 📄 tfidf.py # Feature extraction (TF-IDF) │── 📄 train_model.py # Training ML model │── 📄 evaluate.py # Model evaluation │── 📄 inference.py # Sentiment prediction │── 📄 requirements.txt # Required libraries │── 📊 ulasan_google_play.csv # Scraped dataset │── 📊 cleaned_ulasan_google_play.csv # Preprocessed dataset │── 📊 labeled_ulasan_google_play.csv # Labeled dataset │── 📦 tfidf_vectorizer.pkl # TF-IDF model │── 📦 model_svm.pkl # SVM model

## 🔧 **Installation & Setup**
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-playstore.git
   cd sentiment-analysis-playstore
2. Install dependencies:
   pip install -r requirements.txt

3. Run the Jupyter Notebook:
   jupyter notebook
   
4. Execute each cell in analisis_sentimen.ipynb.

📊 Results
The SVM model achieved over 85% accuracy in sentiment classification.
Example prediction for new review:

"This app crashes frequently and runs very slowly."
Prediction: Negative

🤖 Future Enhancements
Implementing deep learning (LSTM/BERT) for better accuracy.
Expanding dataset with more user reviews.
Adding sentiment trend analysis over time.

📜 License
This project is open-source under the MIT License.
