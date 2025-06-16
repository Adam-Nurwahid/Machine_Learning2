# 📄 Klasifikasi Teks menggunakan RNN (LSTM) - Spam vs Ham

Proyek ini adalah implementasi tugas individu yang bertujuan untuk membangun model klasifikasi teks menggunakan Recurrent Neural Network (RNN), khususnya **Long Short-Term Memory (LSTM)**, untuk membedakan antara pesan **spam** dan **ham** (bukan spam).

---

## 📌 Ringkasan Proyek

- **🎯 Tujuan**: Memprediksi apakah suatu pesan teks termasuk dalam kategori *spam* atau *ham*.
- **📊 Dataset**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) dari Kaggle.
- **🧠 Model**: LSTM dengan lapisan *Embedding* dan *Dropout*.
- **✅ Akurasi Validasi Terbaik**: **90.62%**
- **💻 Bahasa & Framework**: Python dengan TensorFlow/Keras.


## 🧠 Arsitektur Model

Model yang digunakan terdiri dari beberapa *layer* utama untuk memproses teks secara sekuensial:

- **Embedding Layer**: Mengubah token kata menjadi vektor padat dengan dimensi 64. Ukuran kosakata dibatasi hingga 5000 kata.
- **LSTM Layer**: Memproses urutan vektor dengan 64 unit untuk menangkap *dependensi jangka panjang*.
- **Dropout Layer**: Menerapkan regularisasi dengan rate 0.5 untuk mengurangi overfitting.
- **Output Layer**: Menggunakan aktivasi sigmoid untuk klasifikasi biner (*spam* / *ham*).

Contoh kode implementasi arsitektur model:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=50))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()
```

## 📂 Struktur Notebook

Proyek ini disusun dalam satu file notebook Jupyter dengan alur kerja sebagai berikut:

1. **📦 Impor Library**  
   Memuat semua pustaka yang diperlukan seperti `pandas`, `numpy`, `tensorflow`, dan `matplotlib`.

2. **📁 Muat Dataset**  
   Membaca data dari file CSV dan menampilkan statistik awal.

3. **🧹 Pra-pemrosesan Teks**  
   Melakukan:
   - Konversi teks ke huruf kecil
   - Tokenisasi dan padding
   - Encoding label

4. **🏗️ Pembangunan Model LSTM**  
   Mendefinisikan arsitektur jaringan saraf berbasis LSTM.

5. **🎯 Pelatihan dan Visualisasi**  
   Melatih model serta memvisualisasikan kurva akurasi dan loss untuk mendeteksi overfitting/underfitting.

6. **📊 Evaluasi dan Analisis**  
   Mengevaluasi model pada data uji dan menyajikan hasil dalam bentuk *confusion matrix*, akurasi, precision, recall, dan f1-score.

7. **📝 Refleksi Eksperimen**  
   Mencatat tantangan selama pengembangan dan perbaikan model yang dilakukan.

Notebook ini dirancang agar dapat dijalankan dari atas ke bawah secara berurutan untuk mereproduksi hasil eksperimen.


## ⚙️ Cara Menjalankan

Untuk mereproduksi hasil dari proyek ini, ikuti langkah-langkah berikut:

### 1. 📥 Unduh File Proyek
Unduh file notebook `spam-vs-no-spam.ipynb` dan dataset terkait (misalnya `spam.csv`).

### 2. 🛠️ Instalasi Dependensi
Pastikan Anda telah menginstal semua library yang dibutuhkan. Jalankan perintah berikut di terminal atau command prompt:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tensorflow
```
📊 Hasil & Evaluasi
Akurasi Training Akhir: 96.75%

Akurasi Validasi Akhir: 90.62%

Kurva Pelatihan: Kurva akurasi dan loss menunjukkan bahwa model belajar dengan stabil tanpa mengalami overfitting yang parah.

Confusion Matrix: Menunjukkan bahwa prediksi model cukup seimbang antara kelas spam dan ham.

## Riwayat Eksperimen

Tabel berikut merangkum hasil dari beberapa percobaan yang dilakukan:

| Percobaan | Model     | Dropout | Optimizer | Akurasi Validasi | Catatan                        |
|-----------|-----------|---------|-----------|------------------|--------------------------------|
| #1        | LSTM(64)  | 0.0     | Adam      | 65.6%            | Underfitting, pembelajaran lambat |
| #2        | LSTM(64)  | 0.5     | Adam      | 87.5%            | Lebih stabil dan konsisten     |
| #3        | LSTM(64)  | 0.5     | Adam      | 90.62%           | Hasil terbaik, tanpa overfitting |

## 🔍 Refleksi

### 🧩 Tantangan Utama
- Menangani **overfitting** dan **underfitting** selama proses pelatihan.
- Memilih **dimensi embedding** yang optimal untuk representasi kata.
- Menentukan **panjang input (maxlen)** yang paling sesuai untuk data SMS pendek.

### ✅ Solusi yang Diterapkan
- Menggunakan **Dropout** sebagai teknik regularisasi yang efektif.
- Melakukan **visualisasi kurva akurasi dan loss** untuk memantau proses pelatihan.
- Melakukan **eksperimen berulang** dengan memodifikasi struktur dan dimensi layer untuk menemukan konfigurasi terbaik.

### 🤖 Pemanfaatan AI
Saya menggunakan tool **AI generatif (ChatGPT)** untuk membantu dalam beberapa aspek teknis:

- Mendapatkan penjelasan mengenai **arsitektur RNN** dan konsep terkait.
- Menyusun **struktur kode** yang bersih dan terdokumentasi.
- Memberikan panduan dalam proses **evaluasi model**.

🔎 *Catatan:* Semua hasil tetap diverifikasi secara mandiri melalui pengamatan terhadap metrik evaluasi dan grafik performa model.


## 💡 Saran Pengembangan

Untuk meningkatkan performa dan generalisasi model di masa mendatang, berikut beberapa saran pengembangan:

- **📈 Gunakan Dataset yang Lebih Besar**  
  Mengeksplorasi dataset yang lebih besar dan lebih bervariasi dapat membantu model belajar pola yang lebih kompleks dan meningkatkan kemampuan generalisasi.

- **🔁 Coba Arsitektur Lain**  
  Bereksperimen dengan model seperti:
  - **Bi-LSTM** (Bidirectional LSTM)
  - **GRU** (Gated Recurrent Unit)
  - **Transformer-based models** seperti **BERT** (melalui fine-tuning)

- **🧠 Integrasikan Pre-trained Embedding**  
  Menggunakan embedding kata yang telah dilatih sebelumnya seperti:
  - [GloVe](https://nlp.stanford.edu/projects/glove/)
  - [FastText](https://fasttext.cc/)

- **🎯 Lakukan Hyperparameter Tuning**  
  Gunakan pendekatan seperti:
  - **Grid Search**
  - **Random Search**
  - atau **Bayesian Optimization**  
  untuk menemukan kombinasi hyperparameter yang optimal.

Pengembangan lanjutan ini dapat membantu meningkatkan akurasi sekaligus mengurangi risiko overfitting.


## 📚 Referensi

- [SMS Spam Collection Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- François Chollet - *Deep Learning with Python*
- [Dokumentasi Keras](https://keras.io/)
- [Panduan TensorFlow](https://www.tensorflow.org/)

---

## 👤 Author

- **Nama**: Adam Toyib Nur Wahid  
- **Tugas**: Klasifikasi Teks - Tugas Individu
