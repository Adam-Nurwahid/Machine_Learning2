# Generative Adversarial Networks (GAN) - Kelompok 4

## Pendahuluan

Proyek ini mendemonstrasikan implementasi Generative Adversarial Networks (GANs) menggunakan PyTorch. GAN adalah arsitektur jaringan saraf yang terdiri dari dua komponen utama: **Generator** dan **Discriminator**.

- **Generator**: Bertugas membuat data palsu yang menyerupai data asli.
- **Discriminator**: Bertugas membedakan antara data asli dan data buatan Generator.

Keduanya dilatih secara bersamaan dalam permainan minimax:
- Generator belajar menipu Discriminator.
- Discriminator belajar menjadi lebih baik dalam membedakan.

Tujuan akhirnya adalah membuat Generator menghasilkan data yang sangat mirip dengan data asli.

## Lingkungan Pengembangan

Proyek ini dikembangkan menggunakan Python dan PyTorch. Berikut adalah beberapa library utama yang digunakan:

- `torch`: PyTorch library utama
- `torch.nn`: Modul untuk membangun neural networks
- `torch.optim`: Modul untuk optimisasi
- `torchvision`: Untuk dataset dan transformasi gambar
- `matplotlib.pyplot`: Untuk visualisasi data
- `numpy`: Untuk operasi numerik
- `torchvision.utils.make_grid`: Untuk membuat grid gambar

Kode ini dirancang untuk dapat berjalan di GPU (CUDA) jika tersedia, atau fallback ke CPU.

## Arsitektur Model

### Generator

Generator adalah jaringan saraf yang mengambil *noise* acak sebagai input dan menghasilkan gambar. Arsitektur Generator dalam proyek ini menggunakan serangkaian lapisan `Linear` diikuti dengan `BatchNorm1d` dan fungsi aktivasi `LeakyReLU`. Lapisan terakhir menggunakan `Tanh` untuk memastikan output berada dalam rentang [-1, 1], yang sesuai untuk data gambar yang dinormalisasi.

```python
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.Linear(nz, ngf * 4),
            nn.BatchNorm1d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ngf * 4, ngf * 2),
            nn.BatchNorm1d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ngf * 2, ngf),
            nn.BatchNorm1d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(ngf, nc * 28 * 28), # Output image size
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1, 28, 28) # Reshape to image dimensions
