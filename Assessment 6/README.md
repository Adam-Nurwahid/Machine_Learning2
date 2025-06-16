📄 Klasifikasi Teks menggunakan RNN (LSTM) - Spam vs Ham
Proyek ini adalah implementasi tugas individu yang bertujuan untuk membangun model klasifikasi teks menggunakan Recurrent Neural Network (RNN), khususnya Long Short-Term Memory (LSTM), untuk membedakan antara pesan spam dan ham (bukan spam).

📌 Ringkasan Proyek
Tujuan: Memprediksi apakah suatu pesan teks termasuk dalam kategori spam atau bukan.

Dataset: SMS Spam Collection Dataset dari Kaggle

Model: LSTM dengan layer Embedding dan Dropout.

Akurasi Validasi Terbaik: 90.62%

Bahasa & Framework: Python dengan TensorFlow/Keras.

🧠 Arsitektur Model
Model yang digunakan terdiri dari beberapa layer utama untuk memproses teks secara sekuensial:

Embedding Layer: Mengubah token kata menjadi vektor padat dengan dimensi 64. Ukuran kosakata dibatasi hingga 5000 kata.

LSTM Layer: Memproses urutan vektor dengan 64 unit untuk menangkap dependensi jangka panjang.

Dropout Layer: Menerapkan regularisasi dengan rate 0.5 untuk mengurangi overfitting.

Output Layer: Menggunakan aktivasi sigmoid untuk klasifikasi biner (spam/ham).

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=64, input_length=50))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

📂 Struktur Notebook
Proyek ini disusun dalam satu file notebook Jupyter dengan alur kerja sebagai berikut:

Impor Library: Memuat semua pustaka yang diperlukan.

Muat Dataset: Membaca data dari file CSV.

Pra-pemrosesan Teks: Melakukan tokenisasi, padding, dan konversi ke huruf kecil.

Pembangunan Model LSTM: Mendefinisikan arsitektur jaringan saraf.

Pelatihan dan Visualisasi: Melatih model dan memvisualisasikan kurva akurasi serta loss.

Evaluasi dan Analisis: Menganalisis kinerja model menggunakan confusion matrix.

Refleksi Eksperimen: Mencatat tantangan dan solusi selama pengembangan.

⚙️ Cara Menjalankan
Untuk mereproduksi hasil dari proyek ini, ikuti langkah-langkah berikut:

1. Unduh File Proyek
Unduh file notebook spam-vs-no-spam.ipynb dan dataset terkait.

2. Instalasi Dependensi
Pastikan Anda telah menginstal semua library yang dibutuhkan. Buka terminal atau command prompt dan jalankan perintah di bawah ini:

pip install pandas numpy seaborn matplotlib scikit-learn tensorflow

3. Jalankan Notebook
Buka dan jalankan notebook menggunakan Jupyter Notebook, JupyterLab, atau Google Colab.

Untuk Jupyter:

jupyter notebook spam-vs-no-spam.ipynb

Untuk Google Colab:

Buka Google Colab.

Pilih File > Upload notebook... dan unggah file .ipynb.

Jalankan semua sel secara berurutan.

📊 Hasil & Evaluasi
Akurasi Training Akhir: 96.75%

Akurasi Validasi Akhir: 90.62%

Kurva Pelatihan: Kurva akurasi dan loss menunjukkan bahwa model belajar dengan stabil tanpa mengalami overfitting yang parah.

Confusion Matrix: Menunjukkan bahwa prediksi model cukup seimbang antara kelas spam dan ham.

Riwayat Eksperimen
Tabel berikut merangkum hasil dari beberapa percobaan yang dilakukan:

Percobaan

Model

Dropout

Optimizer

Akurasi Validasi

Catatan

#1

LSTM(64)

0.0

Adam

65.6%

Underfitting, pembelajaran lambat

#2

LSTM(64)

0.5

Adam

87.5%

Lebih stabil dan konsisten

#3

LSTM(64)

0.5

Adam

90.62%

Hasil terbaik, tanpa overfitting

🔍 Refleksi
Tantangan Utama
Menangani overfitting dan underfitting selama proses pelatihan.

Memilih dimensi embedding yang optimal untuk representasi kata.

Menentukan panjang input (maxlen) yang paling sesuai untuk data SMS.

Solusi yang Diterapkan
Dropout digunakan sebagai teknik regularisasi yang efektif.

Visualisasi kurva akurasi dan loss sangat membantu dalam memantau dan mengevaluasi proses pelatihan.

Eksperimen berulang dengan memodifikasi struktur dan dimensi layer untuk menemukan konfigurasi terbaik.

Pemanfaatan AI
Saya menggunakan tool AI generatif (ChatGPT) untuk:

Mendapatkan penjelasan teknis mengenai arsitektur RNN dan konsep terkait.

Membantu menyusun struktur kode yang bersih dan terdokumentasi.

Memberikan pandangan dalam proses evaluasi model.

Verifikasi akhir tetap dilakukan dengan membandingkan metrik dan grafik yang dihasilkan secara mandiri.

💡 Saran Pengembangan
Gunakan Dataset yang Lebih Besar: Mengeksplorasi dataset yang lebih besar dan modern untuk meningkatkan generalisasi model.

Coba Arsitektur Lain: Bereksperimen dengan model seperti Bi-LSTM, GRU, atau arsitektur Transformer (misalnya, fine-tuning BERT).

Integrasikan Pre-trained Embedding: Menggunakan word embedding yang sudah dilatih sebelumnya seperti GloVe atau FastText.

Lakukan Hyperparameter Tuning: Menggunakan teknik seperti Grid Search atau Random Search untuk menemukan hyperparameter yang optimal.

📚 Referensi
SMS Spam Collection Dataset - Kaggle

François Chollet - Deep Learning with Python

Dokumentasi Keras

Panduan TensorFlow

👤 Author
Nama: Adam Toyib Nur Wahid

Tugas: Klasifikasi Teks - Tugas Individu
