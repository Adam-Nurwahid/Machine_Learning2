Variational Autoencoder (VAE) pada Dataset Fashion MNIST
Proyek ini mengeksplorasi implementasi Variational Autoencoder (VAE) menggunakan PyTorch untuk mempelajari representasi laten dari dataset Fashion MNIST. VAE adalah model generatif yang mampu melakukan rekonstruksi data input dan menghasilkan sampel data baru dari ruang laten yang terstruktur.

Ringkasan Hasil
Eksperimen ini menunjukkan bahwa model VAE berhasil:

Merekonstruksi Gambar: Model mampu merekonstruksi gambar item pakaian dari dataset Fashion MNIST dengan kualitas yang baik, menunjukkan bahwa model berhasil menangkap fitur-fitur esensial dari data input.

Mempelajari Ruang Laten Bermakna: Melalui visualisasi t-SNE dari ruang laten, terlihat adanya pengelompokan yang jelas untuk kelas-kelas item pakaian yang berbeda. Hal ini menandakan bahwa VAE berhasil mempelajari representasi laten yang terstruktur dan bermakna, di mana gambar-gambar serupa cenderung berdekatan di ruang laten.

Konvergensi Model: Grafik loss menunjukkan penurunan yang signifikan di awal pelatihan dan kemudian stabil mendekati akhir, menandakan bahwa proses pelatihan model telah berhasil konvergen.

Secara keseluruhan, proyek ini mengkonfirmasi kemampuan VAE dalam mempelajari representasi data yang kompleks seperti Fashion MNIST, serta potensi model generatif untuk rekonstruksi dan generasi data.

Petunjuk Eksekusi
Ikuti langkah-langkah di bawah ini untuk menjalankan notebook dan mereplikasi hasil eksperimen.

Prasyarat
Pastikan Anda telah menginstal:

Python 3.8+

pip (manajer paket Python)

Jupyter Notebook atau JupyterLab

Instalasi Dependensi
Kloning Repositori (jika ada):

git clone <URL_GITHUB_REPOSITORI_ANDA>
cd <NAMA_FOLDER_REPOSITORI>

(Jika proyek Anda sudah ada di lokal, lewati langkah ini.)

Buat Virtual Environment (Opsional, tetapi direkomendasikan):

python -m venv venv
# Aktifkan virtual environment
# Di Windows:
# venv\Scripts\activate
# Di macOS/Linux:
# source venv/bin/activate

Instal Paket yang Diperlukan:

pip install torch torchvision matplotlib scikit-learn tqdm jupyter

torch dan torchvision: Untuk membangun dan melatih model VAE, serta mengelola dataset.

matplotlib: Untuk membuat plot visualisasi, seperti grafik loss dan t-SNE.

scikit-learn: Digunakan untuk t-SNE (TSNE) untuk mereduksi dimensi ruang laten.

tqdm: Untuk menampilkan progress bar selama pelatihan.

jupyter: Untuk menjalankan notebook.

Menjalankan Notebook
Buka Jupyter Notebook/Lab:

jupyter notebook

Atau

jupyter lab

Navigasi ke File Notebook:
Di antarmuka Jupyter, navigasikan ke direktori proyek Anda dan buka file vae_fashion_mnist.ipynb.

Jalankan Sel-sel Notebook:
Jalankan setiap sel dalam notebook secara berurutan. Anda bisa menggunakan opsi "Run All" dari menu atau menjalankan sel satu per satu.

Sel Data Loading: Akan mengunduh dataset Fashion MNIST ke dalam folder dataset/.

Sel Pelatihan Model: Akan melatih model VAE selama jumlah epoch yang ditentukan. Anda akan melihat progress bar dan loss di setiap epoch.

Sel Inferensi & Generasi Gambar: Akan menyimpan contoh gambar yang direkonstruksi/dihasilkan ke dalam folder results/.

Sel Visualisasi t-SNE: Akan menghasilkan plot t-SNE dari ruang laten dan menyimpannya sebagai results/latent_space.png. Plot ini juga akan ditampilkan langsung di notebook.

Output yang Diharapkan
Setelah menjalankan seluruh notebook, Anda akan menemukan:

Folder results/: Berisi gambar-gambar yang direkonstruksi/dihasilkan (misalnya, generated_*.png) dan visualisasi ruang laten (latent_space.png).

Plot t-SNE: Tampilan plot 2D dari ruang laten yang menunjukkan klastering kelas-kelas Fashion MNIST.

Konsol Jupyter: Menampilkan progress bar pelatihan dan nilai loss di setiap epoch.

Selamat mencoba!