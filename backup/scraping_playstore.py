#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google_play_scraper import reviews, Sort
import pandas as pd

# Nama paket aplikasi yang akan di-scrape (ganti sesuai kebutuhan)
app_package_name = 'com.gojek.app'
count = 3000  # Jumlah ulasan yang diambil

# Scraping data dari Google Play Store
result, _ = reviews(
    app_package_name,
    lang='id',  # Bahasa Indonesia
    country='id',  # Negara Indonesia
    sort=Sort.NEWEST,  # Ambil ulasan terbaru
    count=count  # Jumlah ulasan
)

# Konversi ke DataFrame
df = pd.DataFrame(result)

# Simpan hasil scraping
df.to_csv('ulasan_google_play.csv', index=False)
print(f"Berhasil mengambil {len(df)} ulasan dan menyimpannya ke 'ulasan_google_play.csv'.")


# In[ ]:




