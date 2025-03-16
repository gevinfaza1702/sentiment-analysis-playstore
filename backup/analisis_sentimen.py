#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re

# Load dataset hasil scraping
df = pd.read_csv('ulasan_google_play.csv')

# Hapus duplikasi
df.drop_duplicates(subset='content', inplace=True)

# Hapus ulasan kosong
df = df[df['content'].str.strip().astype(bool)]

# Membersihkan teks ulasan
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Hanya menyisakan huruf dan spasi
    text = text.lower().strip()  # Mengubah teks menjadi huruf kecil
    return text

df['cleaned_content'] = df['content'].apply(clean_text)

# Simpan hasil preprocessing
df.to_csv('cleaned_ulasan_google_play.csv', index=False)
print(f"Preprocessing selesai! {len(df)} ulasan telah dibersihkan.")


# In[3]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Load dataset yang sudah dibersihkan
df = pd.read_csv('cleaned_ulasan_google_play.csv')

# Pastikan tidak ada nilai NaN
df = df.dropna(subset=['cleaned_content'])

# Konversi ke string untuk mencegah error
df['cleaned_content'] = df['cleaned_content'].astype(str)

# Inisialisasi Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Analisis Sentimen
df['sentiment_score'] = df['cleaned_content'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

# Konversi skor ke kategori sentimen
def label_sentiment(score):
    if score >= 0.05:
        return "positif"
    elif score <= -0.05:
        return "negatif"
    else:
        return "netral"

df['sentiment'] = df['sentiment_score'].apply(label_sentiment)

# Simpan hasil pelabelan
df.to_csv('labeled_ulasan_google_play.csv', index=False)
print("Pelabelan selesai! Data siap untuk analisis sentimen.")


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle

# Load dataset yang sudah memiliki label sentimen
df = pd.read_csv('labeled_ulasan_google_play.csv')

# Ekstraksi fitur menggunakan TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['cleaned_content'])
y = df['sentiment']

# Simpan model TF-IDF agar bisa digunakan nanti
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Ekstraksi fitur dengan TF-IDF selesai!")


# In[5]:


from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle

# Membagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Melatih model SVM
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

# Simpan model
with open("model_svm.pkl", "wb") as f:
    pickle.dump(svm_model, f)

print("Model SVM telah dilatih dan disimpan!")


# In[6]:


from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load model dan data
with open("model_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Prediksi
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print(f"Akurasi Model SVM: {accuracy_svm:.2%}")
print(report_svm)


# In[7]:


from sklearn.model_selection import GridSearchCV
import pickle

# Load dataset dan model
with open("model_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LinearSVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)


# In[8]:


import pickle

# Load model TF-IDF dan model SVM
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model_svm.pkl", "rb") as f:
    svm_model = pickle.load(f)

# Contoh ulasan baru
new_review = ["Aplikasi ini sangat lambat dan sering crash."]
new_review_tfidf = tfidf.transform(new_review)

# Prediksi sentimen
predicted_sentiment = svm_model.predict(new_review_tfidf)

print(f"Prediksi Sentimen: {predicted_sentiment[0]}")


# In[ ]:




