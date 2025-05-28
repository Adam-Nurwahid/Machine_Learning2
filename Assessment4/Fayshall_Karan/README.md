# Deteksi Masker Wajah menggunakan ResNet18

Proyek ini mendemonstrasikan penggunaan model ResNet18 yang telah dilatih sebelumnya (pre-trained) untuk tugas klasifikasi gambar, yaitu mendeteksi apakah seseorang dalam gambar mengenakan masker wajah atau tidak. Notebook ini mencakup seluruh proses mulai dari pemuatan dataset, pra-pemrosesan data, pelatihan model, evaluasi, hingga prediksi pada gambar baru.

## Daftar Isi
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Pra-pemrosesan Data](#pra-pemrosesan-data)
- [Pelatihan Model](#pelatihan-model)
- [Hasil](#hasil)
- [Prediksi](#prediksi)
- [Visualisasi](#visualisasi)
- [Ketergantungan](#ketergantungan)
- [Cara Menjalankan](#cara-menjalankan)

## Dataset
Dataset yang digunakan adalah "Face Mask Detection Dataset" yang bersumber dari Kaggle (`/kaggle/input/face-mask-detection/Dataset`). Dataset ini terdiri dari dua kelas:
- `with_mask`: Gambar orang yang mengenakan masker.
- `without_mask`: Gambar orang yang tidak mengenakan masker.
Gambar-gambar dalam format JPG, JPEG, atau PNG dimuat menggunakan kelas `FaceMaskDataset` yang telah dikustomisasi.

## Arsitektur Model
Model yang digunakan adalah ResNet18, sebuah Convolutional Neural Network (CNN) yang populer, dengan bobot yang telah dilatih sebelumnya pada dataset ImageNet (`ResNet18_Weights.DEFAULT`). Lapisan *fully connected* (fc) terakhir dari ResNet18 dimodifikasi menjadi `nn.Linear` dengan 2 output, sesuai dengan jumlah kelas (dengan masker, tanpa masker).
- Total parameter model: 11.18M
- Parameter yang dapat dilatih: 11.18M

## Pra-pemrosesan Data
Setiap gambar dalam dataset melewati serangkaian transformasi sebelum dimasukkan ke model:
1.  Ukuran gambar diubah menjadi 224x224 piksel.
2.  Gambar dikonversi menjadi Tensor PyTorch.
3.  Normalisasi gambar menggunakan mean `[0.485, 0.456, 0.406]` dan standar deviasi `[0.229, 0.224, 0.225]`.

Dataset kemudian dibagi menjadi data latih (80%) dan data uji (20%). `DataLoader` digunakan untuk memuat data dalam batch berukuran 32.

## Pelatihan Model
- **Device**: Model dilatih menggunakan CUDA (GPU) jika tersedia, jika tidak maka menggunakan CPU.
- **Loss Function**: `nn.CrossEntropyLoss` digunakan sebagai fungsi kerugian.
- **Optimizer**: `optim.Adam` digunakan sebagai optimizer dengan laju pembelajaran (learning rate) `1e-4`.
- **Epoch**: Model dilatih selama 10 epoch.

Selama pelatihan, kerugian (loss) pada data latih dan data uji, serta akurasi pada data uji, dicatat setiap epoch. Model beserta bobot optimizer dan riwayat loss disimpan dalam file `face_mask_checkpoint.pth`.

## Hasil
Setelah 10 epoch pelatihan, model mencapai hasil sebagai berikut:
- **Epoch 10**:
    - Train Loss: 0.0003
    - Test Loss: 0.0009
    - Test Accuracy: 0.9992 (99.92%)

Kerugian pada data latih dan data uji menunjukkan tren penurunan yang baik, mengindikasikan bahwa model belajar dengan efektif.

## Prediksi
Notebook ini juga menyertakan fungsionalitas untuk memuat model yang telah dilatih dan melakukan prediksi pada gambar baru.
- Fungsi `load_model` digunakan untuk memuat checkpoint model.
- Fungsi `predict_image` mengambil path gambar, melakukan pra-pemrosesan yang sama seperti saat pelatihan, dan menghasilkan prediksi kelas (`with_mask` atau `without_mask`).

Contoh prediksi dilakukan pada gambar `/kaggle/input/d/andrewmvd/face-mask-detection/images/maksssksksss15.png`.

## Visualisasi
- **Loss Plot**: Kurva kerugian (loss) data latih dan data uji per epoch divisualisasikan menggunakan Matplotlib untuk memantau proses pembelajaran model.
- **Prediction Display**: Gambar yang diprediksi ditampilkan beserta label prediksinya menggunakan Matplotlib.

## Ketergantungan
Proyek ini menggunakan pustaka Python berikut:
- `torch`
- `torchvision`
- `os`
- `glob`
- `PIL (Pillow)`
- `tqdm`
- `matplotlib`

## Cara Menjalankan
1.  **Persiapan Dataset**: Pastikan dataset "Face Mask Detection" tersedia pada path yang sesuai (misalnya, `/kaggle/input/face-mask-detection/Dataset`).
2.  **Instalasi Ketergantungan**: Instal semua pustaka yang tercantum di atas.
3.  **Menjalankan Notebook**:
    - Untuk melatih model dari awal, jalankan semua sel notebook secara berurutan.
    - Untuk melakukan prediksi menggunakan model yang telah dilatih (pastikan file `face_mask_checkpoint.pth` ada):
        - Jalankan sel-sel yang berisi definisi fungsi `load_model` dan `predict_image`.
        - Modifikasi `img_path` pada bagian `if __name__ == '__main__':` untuk menunjuk ke gambar yang ingin diprediksi.
        - Jalankan sel tersebut untuk melihat hasil prediksi dan visualisasi gambar.
