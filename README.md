# Klasifikasi Bunga Iris dengan Multilayer Perceptron (MLP)

## 📚 Deskripsi Proyek

Proyek ini merupakan bagian dari tugas kelompok untuk mata kuliah **[Nama Mata Kuliah]**, yang bertujuan untuk mengimplementasikan algoritma **Multilayer Perceptron (MLP)** menggunakan PyTorch untuk klasifikasi bunga **Iris**. Dataset yang digunakan adalah dataset Iris dari Scikit-learn.

Kami melakukan eksplorasi data, membangun model MLP, melakukan pelatihan model, mengevaluasi performa, dan membandingkannya dengan model sederhana seperti **Logistic Regression**.

---

## 🎯 Tujuan Pembelajaran

- Menjelaskan fungsi aktivasi dan peranannya dalam deep learning.
- Mengimplementasikan MLP menggunakan PyTorch.
- Membandingkan performa model MLP dengan Logistic Regression.
- Menyusun laporan ilmiah berdasarkan eksperimen dan analisis kode.
- Menjelaskan potensi aplikasi dan tantangan penerapan model di dunia nyata.

---

## 🛠 Tools dan Library

- Python 3.8+
- PyTorch
- Scikit-learn
- Pandas
- Matplotlib
- Jupyter Notebook

---

## 🗂 Struktur Proyek

## 🗂 Struktur Proyek

Berikut adalah struktur direktori dari proyek klasifikasi bunga Iris dengan MLP:

.
├── data/
│ └── iris.csv # Dataset Iris (jika disimpan manual, opsional)
│
├── model.py # Arsitektur jaringan MLP (PyTorch Module)
├── train.py # Skrip pelatihan model MLP
├── predict.py # Skrip prediksi untuk data baru menggunakan model.pth
├── model.pth # Model hasil pelatihan yang disimpan
│
├── iris_logreg.py # Implementasi model Logistic Regression untuk perbandingan
│
├── utils.py # Fungsi-fungsi bantu (plot, preprocessing, dll) (opsional)
├── requirements.txt # Daftar dependensi Python yang digunakan
│
├── laporan_akhir.ipynb # Laporan akhir dalam bentuk notebook Jupyter
├── laporan_akhir.pdf # Laporan akhir dalam format PDF
├── README.md # Dokumentasi proyek ini (yang sedang Anda baca)
│
└── assets/
├── training_loss.png # Visualisasi loss selama training
├── training_accuracy.png # Visualisasi akurasi selama training
└── confusion_matrix.png # Confusion matrix hasil evaluasi model
