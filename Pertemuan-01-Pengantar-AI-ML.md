# PERTEMUAN 1 — Pengantar AI & ML: Sejarah, Paradigma, dan Etika

Mata Kuliah: Machine Learning & AI · Durasi: 150 menit · Format: Kuliah interaktif + demo live

---

## 1. Metadata Pertemuan

**Tujuan Pembelajaran (Capaian Pembelajaran):**

1. Mahasiswa mampu **mendefinisikan** AI, ML, dan Deep Learning serta menjelaskan relasi hierarkisnya dengan tepat, merujuk pada definisi Russell-Norvig dan Mitchell.
2. Mahasiswa mampu **mengklasifikasikan** sebuah masalah nyata ke dalam paradigma yang sesuai (supervised / unsupervised / reinforcement learning) beserta justifikasinya.
3. Mahasiswa mampu **menjalankan** pipeline ML end-to-end sederhana (load data → split → train → evaluate) menggunakan scikit-learn di Google Colab.
4. Mahasiswa mampu **mengidentifikasi** minimal 3 risiko etis sistem ML (bias, privasi, akuntabilitas) pada studi kasus nyata.

**Prasyarat:**

- Pemrograman Python dasar (variabel, fungsi, list, import library).
- Statistika deskriptif dasar (mean, varians) — akan diperdalam di Pertemuan 2.
- Tidak ada prasyarat ML; ini pertemuan pembuka.

**Estimasi Waktu: 150 menit**

| Segmen | Durasi |
|---|---|
| Hook + Sejarah & definisi (Slide 1–6) | 35 menit |
| Paradigma & anatomi ML (Slide 7–16) | 45 menit |
| Etika (Slide 17–19) | 20 menit |
| Studi kasus + demo live (Slide 20–24) | 35 menit |
| Rangkuman + briefing tugas (Slide 25–27) | 15 menit |

---

## 2. Outline PPT — 27 Slide

### FASE HOOK (1 slide)

**[Slide 1] Mesin di Sekitar Anda Sudah "Belajar"**
- Pagi ini: filter spam menyaring email Anda tanpa aturan eksplisit.
- Perjalanan ke kampus: Google Maps memprediksi waktu tempuh dari data jutaan pengguna.
- Membuka ponsel: face unlock, rekomendasi video, autocorrect — semuanya ML.
- Pertanyaan pembuka: *bagaimana mungkin mesin melakukan ini tanpa diprogram aturan per kasus?*
- *Catatan Pengajar:* Mulai dengan polling cepat — minta mahasiswa sebutkan 3 aplikasi ML yang mereka pakai hari ini sebelum menampilkan bullet. Tujuannya membangun rasa bahwa ML bukan topik futuristik, melainkan infrastruktur sehari-hari.
- *Visual:* Kolase 4 screenshot aplikasi sehari-hari (Gmail spam, Maps, rekomendasi YouTube, face unlock) dengan tanda tanya besar di tengah.

### FASE KONSEP INTI (18 slide)

**[Slide 2] Apa Itu Artificial Intelligence? Empat Kuadran Russell-Norvig**
- AI didefinisikan lewat 2 sumbu: *berpikir vs bertindak* × *seperti manusia vs secara rasional*.
- Thinking humanly → cognitive science; Acting humanly → Turing Test.
- Thinking rationally → logika formal; Acting rationally → **rational agent** (definisi modern).
- Buku ini memakai definisi rational agent: agen yang bertindak memaksimalkan ukuran kinerja.
- *Catatan Pengajar:* Tekankan bahwa definisi "acting rationally" adalah yang dipakai riset modern karena terukur dan tidak bergantung meniru manusia (AIMA Bab 1.1). Turing Test menarik secara historis tetapi bukan target engineering saat ini.
- *Visual:* Matriks 2×2 empat kuadran definisi AI (AIMA Bab 1).

**[Slide 3] Sejarah AI (1950–1980): Optimisme Simbolik**
- 1950: Alan Turing, *"Computing Machinery and Intelligence"* — imitation game.
- 1956: Dartmouth Workshop — istilah "Artificial Intelligence" lahir (McCarthy dkk.).
- Era GOFAI: kecerdasan = manipulasi simbol & aturan logika (expert systems).
- Keberhasilan awal di dunia mikro (checkers, teorema logika), gagal di dunia nyata terbuka.
- *Catatan Pengajar:* Ceritakan bahwa peneliti 1960-an memprediksi mesin secerdas manusia dalam 20 tahun — pelajaran tentang hype yang berulang hingga kini. Sambungkan ke diskusi akhir tentang klaim berlebihan pada AI modern (AIMA Bab 1.3).
- *Visual:* Timeline horizontal 1950–1980 dengan foto Turing dan peserta Dartmouth.

**[Slide 4] AI Winter dan Kebangkitan Machine Learning (1980–2010)**
- Dua "AI Winter" (±1974, ±1987): dana riset anjlok karena janji tak terpenuhi.
- Pergeseran paradigma: dari aturan ditulis manusia → **pola dipelajari dari data**.
- 1990-an–2000-an: SVM, decision tree, ensemble mendominasi; ML jadi disiplin statistik.
- Pemicu kebangkitan: data digital melimpah + komputasi murah + algoritma matang.
- *Catatan Pengajar:* Kontraskan expert system (ribuan aturan if-then ditulis tangan) dengan classifier yang menemukan aturannya sendiri dari contoh. Ini transisi konseptual terpenting dalam sejarah bidang ini (Bishop Bab 1; Goodfellow Bab 1.2).
- *Visual:* Grafik garis "minat & pendanaan AI" naik-turun dengan dua lembah AI Winter.

**[Slide 5] Era Deep Learning (2012–Sekarang)**
- 2012: AlexNet menang ImageNet — error turun drastis dari 26% ke 16% dalam satu tahun.
- 2016: AlphaGo mengalahkan Lee Sedol; 2017: arsitektur Transformer lahir.
- 2020-an: LLM (GPT, Gemini, Claude) — model generatif skala miliaran parameter.
- Resep era ini: big data + GPU + neural network dalam (deep).
- *Catatan Pengajar:* Goodfellow Bab 1.2 mencatat bahwa ide neural network sudah ada sejak 1940-an; yang baru adalah skala data dan komputasi — bukan konsepnya. Ini melawan miskonsepsi bahwa DL adalah penemuan mendadak.
- *Visual:* Grafik error ImageNet 2010–2017 menurun tajam, dianotasi titik AlexNet.

**[Slide 6] AI ⊃ ML ⊃ Deep Learning: Peta Wilayah**
- AI: payung besar — semua teknik membuat mesin bertindak cerdas (termasuk logika, search, planning).
- ML: subhimpunan AI — sistem yang **meningkat kinerjanya dari pengalaman/data**.
- Deep Learning: subhimpunan ML — representasi dipelajari berlapis via neural network.
- Konsekuensi: ada AI tanpa ML (rule-based chess), ada ML tanpa DL (random forest).
- *Catatan Pengajar:* Gambar diagram Venn ini di papan dan uji dengan contoh: "Sistem pakar diagnosis dengan 500 aturan — masuk mana?" (AI, bukan ML). Diagram serupa ada di Goodfellow Bab 1, Gambar 1.4.
- *Visual:* Diagram Venn tiga lingkaran bersarang AI > ML > DL dengan contoh teknik di tiap lapisan.

**[Slide 7] Definisi Formal ML: Tugas, Pengalaman, Kinerja**
- Definisi Tom Mitchell: program *belajar* dari pengalaman **E** terhadap tugas **T** dengan ukuran kinerja **P**, jika kinerja pada T (diukur P) membaik seiring E.
- Contoh spam: T = klasifikasi email, E = ribuan email berlabel, P = akurasi.
- Definisi ini operasional: memaksa kita menspesifikasikan T, E, P sebelum coding.
- Latihan kilat: definisikan T/E/P untuk prediksi harga rumah.
- *Catatan Pengajar:* Minta 2–3 mahasiswa mendefinisikan T/E/P untuk kasus berbeda (deteksi wajah, rekomendasi film) — kesalahan umum adalah mencampur T dengan P. Definisi ini dibahas di Goodfellow Bab 5.1.
- *Visual:* Kartu tiga kolom T / E / P dengan contoh spam terisi.

**[Slide 8] Paradigma 1 — Supervised Learning: Belajar dari Label**
- Data: pasangan input–output $(x_i, y_i)$; tujuan: pelajari pemetaan $f: x \mapsto y$.
- Dua tugas utama: **klasifikasi** (y diskret: spam/bukan) dan **regresi** (y kontinu: harga).
- Analogi: belajar dengan kunci jawaban — guru (label) mengoreksi setiap contoh.
- Mayoritas aplikasi industri saat ini adalah supervised.
- *Catatan Pengajar:* Tekankan biaya tersembunyi: label mahal karena butuh anotasi manusia — inilah alasan ekonomi di balik riset unsupervised/self-supervised (Bishop Bab 1.1; Géron Bab 1). Beri contoh biaya anotasi medis oleh radiolog.
- *Visual:* Diagram alir: dataset berlabel → training → model → prediksi pada data baru.

**[Slide 9] Kode #1: Supervised Learning dalam 8 Baris**
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)          # E: 150 bunga berlabel
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_tr, y_tr)                       # T: klasifikasi spesies
print(model.score(X_te, y_te))              # P: akurasi ≈ 0.98
```
- Perhatikan: tidak ada satu pun aturan "if petal > 2cm then ..." yang kita tulis.
- Komentar kode memetakan langsung ke T/E/P Mitchell.
- *Catatan Pengajar:* Jangan jelaskan cara kerja K-NN dulu (itu Pertemuan 4) — fokus pada *bentuk* pipeline: fit lalu score. Jalankan live di Colab agar mahasiswa melihat angka akurasi muncul nyata.
- *Visual:* Blok kode dengan 3 anotasi panah berwarna menunjuk E, T, P.

**[Slide 10] Paradigma 2 — Unsupervised Learning: Struktur Tanpa Label**
- Data: hanya $x_i$, tanpa jawaban benar; tujuan: temukan struktur tersembunyi.
- Clustering: kelompokkan pelanggan serupa; Dimensionality reduction: padatkan fitur (PCA).
- Anomaly detection: transaksi kartu kredit yang "aneh" dibanding pola normal.
- Analogi: menyortir tumpukan foto tanpa diberi tahu kategorinya.
- *Catatan Pengajar:* Tanyakan "bagaimana mengukur benar/salah tanpa label?" — biarkan menggantung sebagai motivasi Pertemuan 6 dan 7. Bishop Bab 1.1 dan Géron Bab 1 memberi taksonomi lengkapnya.
- *Visual:* Scatter plot titik tanpa warna → sesudah clustering menjadi 3 warna kelompok.

**[Slide 11] Paradigma 3 — Reinforcement Learning: Belajar dari Konsekuensi**
- Agen berinteraksi dengan lingkungan: aksi → state baru + **reward**.
- Tidak ada dataset statis; data dihasilkan dari pengalaman mencoba (trial-and-error).
- Contoh: AlphaGo, robot berjalan, pengaturan pendingin data center Google.
- Tantangan khas: reward tertunda — langkah ke-10 menentukan kalah/menang di langkah ke-100.
- *Catatan Pengajar:* Bedakan tegas dari supervised: tidak ada "jawaban benar per langkah", hanya sinyal reward yang jarang dan tertunda (AIMA Bab 22; detail di Pertemuan 13). Analogi melatih anjing dengan camilan biasanya efektif di kelas.
- *Visual:* Diagram loop agen ↔ lingkungan (state, action, reward).

**[Slide 12] Taksonomi Tugas ML: Peta Cepat**
- Klasifikasi (spam?), Regresi (harga?), Clustering (segmen pelanggan?).
- Dimensionality reduction (kompresi fitur), Anomaly detection (fraud), Generation (buat gambar/teks).
- Satu masalah bisnis sering = kombinasi beberapa tugas.
- Kuis kilat: "prediksi churn pelanggan" → tugas apa? (klasifikasi biner).
- *Catatan Pengajar:* Slide ini adalah peta 16 pertemuan ke depan — tunjukkan pertemuan berapa membahas tugas mana agar mahasiswa punya gambaran arah kuliah. Kuis kilat 3 soal via angkat tangan menjaga energi kelas.
- *Visual:* Tabel 6 tugas × kolom (contoh input, contoh output, pertemuan terkait).

**[Slide 13] Anatomi Proyek ML: Pipeline End-to-End**
- Alur baku: **Data → Preprocessing → Fitur → Training → Evaluasi → Deployment**.
- 60–80% waktu praktisi habis di data & preprocessing, bukan pemodelan.
- Pipeline bersifat siklus: hasil evaluasi memicu revisi data/fitur (iteratif).
- Pertemuan 14 (MLOps) membahas otomasi seluruh siklus ini.
- *Catatan Pengajar:* Géron Bab 2 memakai proyek harga rumah California untuk seluruh pipeline ini — sebutkan sebagai bacaan lanjutan bagi yang penasaran. Tekankan realita industri: model hanyalah komponen kecil dari sistem ML produksi.
- *Visual:* Diagram alir pipeline 6 tahap dengan panah balik dari Evaluasi ke Data.

**[Slide 14] Model = Fungsi Berparameter**
- Model adalah fungsi matematis: $\hat{y} = f(x; \theta)$ — input $x$, parameter $\theta$, prediksi $\hat{y}$.
- "Belajar" = mencari $\theta$ yang meminimalkan **loss**: $\theta^* = \arg\min_\theta \; \frac{1}{N}\sum_{i=1}^{N} L\big(f(x_i;\theta),\, y_i\big)$
- Contoh: regresi linear $f(x;\theta) = \theta_0 + \theta_1 x$ — belajar = menggeser garis.
- Seluruh ML (dari regresi sampai LLM) adalah variasi dari kerangka ini.
- *Catatan Pengajar:* Ini slide matematis pertama — bacakan notasinya pelan-pelan dalam bahasa lisan ("theta adalah kenop yang diputar-putar sampai kesalahan minimal"). Kerangka loss-minimization ini formal di Goodfellow Bab 5.1–5.2 dan menjadi tulang punggung Pertemuan 3.
- *Visual:* Animasi/3-frame garis regresi yang berputar mendekati sebaran titik, dengan nilai loss menurun.

**[Slide 15] Generalisasi: Ujian Sesungguhnya Ada di Data Baru**
- Tujuan ML bukan menghafal data latih, tetapi berkinerja baik pada **data belum terlihat**.
- Overfitting: nilai sempurna di latihan, gagal di ujian — model menghafal noise.
- Underfitting: model terlalu sederhana, gagal di keduanya.
- Karena itu data WAJIB dipisah: train set vs test set (test = "soal ujian tersegel").
- *Catatan Pengajar:* Analogi paling efektif: mahasiswa yang menghafal soal tahun lalu vs yang memahami konsep — ujiannya soal baru. Konsep kapasitas/generalisasi diformalkan di Goodfellow Bab 5.2 dan Bishop Bab 1.3; dibahas dalam di Pertemuan 7.
- *Visual:* Dua plot berdampingan: kurva mulus (pas) vs kurva berliku ekstrem melewati semua titik (overfit).

**[Slide 16] Kode #2: Disiplin Train/Test Split**
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=None)   # bebas tumbuh → rawan overfit
model.fit(X_tr, y_tr)
print("Akurasi train:", model.score(X_tr, y_tr))  # 1.00  ← curiga!
print("Akurasi test :", model.score(X_te, y_te))  # 0.93  ← angka jujur
```
- Akurasi train 100% + test lebih rendah = sinyal overfitting.
- `random_state` membuat eksperimen reproducible; `stratify` menjaga proporsi kelas.
- *Catatan Pengajar:* Jalankan live dan tunjukkan gap train vs test — ini bukti empiris overfitting pertama yang mahasiswa lihat sendiri. Tegaskan aturan emas: test set tidak boleh disentuh sampai evaluasi akhir (kebocoran data dibahas di Pertemuan 7).
- *Visual:* Output kode dengan dua angka akurasi di-highlight kontras (hijau vs kuning).

**[Slide 17] Etika #1 — Bias: Model Mewarisi Ketimpangan Data**
- Model belajar dari data historis; jika data mengandung diskriminasi, model mereplikasinya.
- Contoh nyata: sistem rekrutmen yang menghukum CV berkata "women's" karena data historis didominasi pria.
- Bias masuk dari: sampling tak representatif, label subjektif, proxy variable (kode pos ≈ ras).
- "Algoritma itu netral" adalah mitos — netralitas data harus dibuktikan, bukan diasumsikan.
- *Catatan Pengajar:* Kasus tool rekrutmen Amazon (dihentikan 2018) adalah contoh terdokumentasi baik untuk dibahas 3 menit. AIMA Bab 27 membahas fairness formal; tekankan bahwa menghapus kolom sensitif TIDAK cukup karena ada proxy.
- *Visual:* Diagram alir: data historis bias → training → keputusan bias → memperkuat data bias (feedback loop).

**[Slide 18] Etika #2 — Privasi, Transparansi, Akuntabilitas**
- Privasi: model bisa "membocorkan" data latih (memorization) — sensitif untuk data medis/finansial.
- Transparansi: banyak model = black box; pasien/nasabah berhak atas penjelasan keputusan.
- Akuntabilitas: jika mobil otonom celaka atau kredit ditolak keliru — siapa bertanggung jawab?
- Trade-off nyata: model paling akurat sering paling sulit dijelaskan.
- *Catatan Pengajar:* Angkat konteks Indonesia: UU PDP No. 27/2022 mengatur pemrosesan data pribadi — relevan langsung bagi calon praktisi ML lokal. Diskusi akuntabilitas sengaja dibiarkan terbuka; tidak ada jawaban tunggal (AIMA Bab 27).
- *Visual:* Tiga ikon (gembok, kotak hitam bertanda tanya, palu hakim) dengan satu kalimat kunci per ikon.

**[Slide 19] Lanskap Regulasi & Praktik Bertanggung Jawab**
- EU AI Act (berlaku bertahap sejak 2024): klasifikasi risiko — unacceptable / high / limited / minimal.
- Indonesia: UU PDP 27/2022; SE Menkominfo No. 9/2023 tentang etika AI (panduan awal).
- Praktik baik industri: model cards, datasheets for datasets, audit bias berkala.
- Sebagai engineer: etika bukan pelengkap di akhir, tetapi keputusan desain sejak awal.
- *Catatan Pengajar:* Tidak perlu hafal pasal — tujuan slide ini menunjukkan bahwa regulasi nyata sudah mengikat pekerjaan ML engineer. Kaitkan dengan Tugas B yang meminta audit bias sederhana.
- *Visual:* Piramida 4 tingkat risiko EU AI Act dengan contoh aplikasi per tingkat.

### FASE STUDI KASUS / IMPLEMENTASI (5 slide)

**[Slide 20] Studi Kasus Etika: COMPAS — Prediksi Residivisme**
- COMPAS: sistem skor risiko residivisme yang dipakai pengadilan AS.
- Investigasi ProPublica (2016): false positive rate terdakwa kulit hitam ≈ 2× lipat kulit putih.
- Pembelaan vendor: skor terkalibrasi sama antar ras — dua definisi "adil" yang terbukti tak bisa dipenuhi bersamaan.
- Pelajaran: memilih metrik fairness adalah keputusan etis, bukan sekadar teknis.
- *Catatan Pengajar:* Ini contoh terbaik bahwa "fairness" punya definisi matematis ganda yang saling bertentangan (impossibility result Kleinberg dkk. 2016). Diskusikan 5 menit: metrik mana yang akan mahasiswa pilih jika jadi hakimnya?
- *Visual:* Bar chart perbandingan false positive & false negative rate antar kelompok (data ProPublica).

**[Slide 21] Studi Kasus Teknis: Klasifikasi Iris End-to-End — Setup**
- Masalah: prediksi spesies bunga iris dari 4 ukuran fisik (T = klasifikasi 3 kelas).
- Data: 150 sampel berlabel klasik dari Fisher, 1936 (E).
- Metrik: akurasi pada 30% data test yang disisihkan (P).
- Alat: Google Colab (gratis, tanpa instalasi) + scikit-learn.
- *Catatan Pengajar:* Framing-kan eksplisit dengan template T/E/P dari Slide 7 agar konsisten — mahasiswa melihat teori langsung dipakai. Pastikan semua sudah bisa membuka Colab sebelum lanjut; ini juga cek kesiapan Tugas A.
- *Visual:* Foto 3 spesies iris + tabel 5 baris pertama dataset (sepal/petal length/width).

**[Slide 22] Studi Kasus Teknis: Kode Lengkap & Hasil**
```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                          random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
print(classification_report(y_te, model.predict(X_te),
                            target_names=iris.target_names))
```
- Satu pipeline utuh: data → split → train → laporan metrik per kelas.
- Output: precision/recall/f1 per spesies — lebih informatif dari sekadar akurasi.
- *Catatan Pengajar:* Ini inti demo live 10 menit (lihat Bagian 5) — ketik atau jalankan sel per sel, jangan tampilkan sekaligus. Belum perlu menjelaskan matematika logistic regression (Pertemuan 3); fokus ke alur dan cara membaca output.
- *Visual:* Screenshot output classification_report + heatmap confusion matrix 3×3.

**[Slide 23] Membaca Hasil: Apa yang Model Ini TIDAK Tahu**
- Akurasi 97% ≠ selesai: model hanya valid untuk distribusi serupa data latih.
- Beri foto mawar → model tetap menjawab salah satu dari 3 iris (tidak bisa bilang "bukan iris").
- Confusion matrix menunjukkan kelas mana yang tertukar (versicolor ↔ virginica).
- Batasan ini memotivasi: evaluasi mendalam (P7), deteksi out-of-distribution, kalibrasi.
- *Catatan Pengajar:* Demo-kan langsung: prediksi input ngawur `model.predict([[9.9, 9.9, 9.9, 9.9]])` dan tunjukkan model tetap percaya diri menjawab — momen "aha" tentang batas ML. Ini menanamkan skeptisisme sehat sejak pertemuan pertama.
- *Visual:* Confusion matrix dianotasi + kotak peringatan "model tidak tahu bahwa ia tidak tahu".

**[Slide 24] Ekosistem & Perkakas Kuliah Ini**
- Bahasa: Python 3.10+; lingkungan: Google Colab (utama) / Jupyter lokal.
- Library inti: NumPy, pandas, matplotlib, scikit-learn; nanti: TensorFlow/Keras & PyTorch (P8+).
- Sumber data latihan: scikit-learn datasets, Kaggle, UCI ML Repository.
- Kontrol versi tugas: GitHub — semua tugas dikumpulkan sebagai repo/notebook.
- *Catatan Pengajar:* Minta mahasiswa membuat akun Kaggle dan GitHub minggu ini juga karena Tugas A–B memakainya. Géron Bab 1–2 adalah panduan setup terbaik untuk pemula; bagikan link environment Colab template kelas.
- *Visual:* Grid logo tools dikelompokkan: bahasa / library / data / kolaborasi.

### FASE RANGKUMAN (1 slide)

**[Slide 25] Rangkuman: Peta yang Anda Bawa Pulang**
- AI ⊃ ML ⊃ DL; ML = kinerja pada tugas T membaik dari pengalaman E diukur P (Mitchell).
- Tiga paradigma: supervised (berlabel), unsupervised (tanpa label), reinforcement (reward).
- Model = fungsi berparameter $f(x;\theta)$; belajar = minimalkan loss; ujian sejati = data baru.
- Etika (bias, privasi, akuntabilitas) adalah keputusan desain, bukan renungan belakangan.
- Minggu depan: matematika yang membuat semua ini bekerja (aljabar linear, kalkulus, probabilitas).
- *Catatan Pengajar:* Tutup dengan kembali ke pertanyaan hook Slide 1 — sekarang mahasiswa bisa menjawabnya dengan kosakata teknis. Beri teaser konkret P2: "kita akan lihat mengapa turunan adalah mesin di balik kata 'belajar'".
- *Visual:* Mind map satu halaman menghubungkan semua konsep pertemuan ini.

### FASE TUGAS (2 slide)

**[Slide 26] Tugas A — Notebook Pertamamu: Iris + Satu Dataset Pilihan**
- Reproduksi pipeline Iris hari ini di Colab, lalu ulangi pada 1 dataset lain (Wine / Breast Cancer dari sklearn).
- Wajib ada: EDA singkat (5 visualisasi), train/test split, model, classification report, sel Markdown berisi definisi T/E/P.
- Kumpulkan: `nim_nama_tugas1a.ipynb` via GitHub repo pribadi.
- Deadline: sebelum Pertemuan 2. Estimasi: 3 jam.
- *Catatan Pengajar:* Tunjukkan contoh struktur notebook yang baik (judul, sel markdown penjelasan, kode bersih) — standar ini dipakai sampai pertemuan 16. Ingatkan bahwa menyalin tanpa memahami akan terlihat saat kuis awal P2.
- *Visual:* Screenshot notebook contoh dengan anotasi bagian-bagian wajib.

**[Slide 27] Tugas B — Audit Mini: Menemukan Bias di Dataset Nyata**
- Analisis dataset Adult Income (UCI): hitung proporsi label income >50K per gender & per ras memakai pandas.
- Tulis temuan dalam laporan 2 halaman: 3 visualisasi + risiko jika dataset ini dipakai melatih model kredit.
- Kumpulkan: `nim_nama_tugas1b.ipynb` + `laporan.pdf` (2 halaman).
- Deadline: sebelum Pertemuan 2. Estimasi: 3 jam.
- *Catatan Pengajar:* Tegaskan bahwa tugas ini analisis data + argumen, bukan esai opini — setiap klaim harus didukung angka dari notebook. Kaitkan kembali ke COMPAS: mahasiswa kini melakukan audit versi sederhana seperti jurnalis ProPublica.
- *Visual:* Contoh bar chart proporsi income per kelompok demografis dari dataset Adult.

---

## 3. Dua Tugas Mandiri Output-Based

### Tugas A: Notebook Pertamamu — Pipeline Klasifikasi dari Contoh ke Mandiri

**Deskripsi:** Mahasiswa mereproduksi pipeline klasifikasi Iris dari kelas, lalu menerapkannya secara mandiri pada satu dataset lain (Wine atau Breast Cancer Wisconsin dari `sklearn.datasets`). Notebook harus memuat EDA (minimal 5 visualisasi), definisi eksplisit T/E/P dalam sel Markdown, train/test split yang benar, dan interpretasi classification report dalam kalimat sendiri.

**Output yang dikumpulkan:** `nim_nama_tugas1a.ipynb` di repo GitHub pribadi (link repo disubmit ke LMS). Notebook harus bisa dijalankan Runtime > Run all tanpa error.

**Rubrik Penilaian Singkat:**

1. **Kebenaran teknis (40%)** — pipeline berjalan tanpa error, split dilakukan sebelum training, tidak ada kebocoran test set.
2. **Kualitas analisis (35%)** — T/E/P didefinisikan tepat, EDA relevan (bukan asal plot), interpretasi metrik per kelas benar.
3. **Keterbacaan (25%)** — struktur notebook rapi (judul, markdown penjelas antar sel), kode berkomentar, penamaan variabel jelas.

**Estimasi pengerjaan:** 3 jam.

### Tugas B: Audit Mini Bias Dataset — Adult Income

**Deskripsi:** Mahasiswa melakukan audit bias sederhana pada dataset Adult Income (UCI): memuat data dengan pandas, menghitung distribusi label pendapatan (>50K) menurut gender dan ras, memvisualisasikan disparitasnya, lalu menulis analisis risiko 2 halaman — apa yang bisa salah jika dataset ini dipakai melatih model persetujuan kredit, dan mitigasi apa yang mereka usulkan.

**Output yang dikumpulkan:** `nim_nama_tugas1b.ipynb` (perhitungan + minimal 3 visualisasi) dan `nim_nama_laporan1b.pdf` (maksimal 2 halaman) via repo GitHub yang sama.

**Rubrik Penilaian Singkat:**

1. **Kebenaran perhitungan (35%)** — agregasi pandas benar, proporsi dihitung per kelompok (bukan angka mentah yang menyesatkan).
2. **Ketajaman analisis risiko (40%)** — mengaitkan temuan angka dengan konsep bias/proxy variable dari kelas, argumen didukung data, mitigasi masuk akal.
3. **Komunikasi (25%)** — visualisasi berjudul dan berlabel sumbu, laporan padat 2 halaman, bahasa akademik komunikatif.

**Estimasi pengerjaan:** 3 jam.

**Catatan progresivitas:** Pertemuan 1 fokus pada *menjalankan dan membaca* pipeline (belum membangun algoritma dari nol — itu mulai Pertemuan 3). Level tugas akan naik bertahap: implementasi dari nol (P3–P8) → arsitektur deep learning (P9–P13) → proyek mini terintegrasi (P14–P16).

---

## 4. Daftar Pustaka & Referensi Tambahan

**Buku utama (bab yang relevan untuk pertemuan ini):**

1. Russell, S. & Norvig, P. *Artificial Intelligence: A Modern Approach* (4th ed.) — **Bab 1** (Introduction: definisi AI, sejarah, state of the art) dan **Bab 27** (Philosophy, Ethics, and Safety of AI). Rujukan utama untuk definisi AI dan etika.
2. Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning* — **Bab 1** (Introduction: sejarah gelombang neural network, Gambar 1.4 hierarki AI/ML/DL) dan **Bab 5.1–5.2** (Learning Algorithms: definisi T/E/P, kapasitas & generalisasi). Rujukan utama untuk kerangka formal ML.
3. Bishop, C. *Pattern Recognition and Machine Learning* — **Bab 1.1–1.3** (Introduction: contoh polynomial curve fitting, taksonomi supervised/unsupervised, model selection & generalisasi).
4. Géron, A. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.) — **Bab 1** (The Machine Learning Landscape: taksonomi paradigma dengan contoh kode) dan **Bab 2** (End-to-End Machine Learning Project: pipeline lengkap). Rujukan utama sisi praktis/implementasi.

**Paper & artikel tambahan:**

5. LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep Learning." *Nature*, 521, 436–444. — Survei ringkas era deep learning oleh tiga pionirnya; bacaan pendamping Slide 5.
6. Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). "Machine Bias." *ProPublica*; dilengkapi Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). "Inherent Trade-Offs in the Fair Determination of Risk Scores." *arXiv:1609.05807*. — Dasar studi kasus COMPAS (Slide 20) dan hasil ketakmungkinan dua definisi fairness dipenuhi sekaligus.

---

## 5. Catatan Pengajar

### Kesalahpahaman umum mahasiswa (siapkan counter-nya)

1. **"AI = robot / AI = ChatGPT."** Luruskan dengan diagram Venn Slide 6: robotika dan LLM hanyalah dua aplikasi; mayoritas ML produksi adalah tabel dan angka (kredit, fraud, rekomendasi).
2. **"Model belajar sendiri tanpa campur tangan manusia."** Tunjukkan bahwa manusia memilih data, label, fitur, metrik, dan threshold — "belajar" hanyalah optimasi $\theta$ dalam kerangka yang sepenuhnya didesain manusia.
3. **"Akurasi tinggi = model bagus."** Gunakan gap train/test dari Slide 16 dan kasus kelas tak seimbang (akurasi 99% pada data fraud 1% adalah model yang tak berguna) — pintu masuk ke Pertemuan 7.
4. **"Algoritma itu objektif karena matematika."** COMPAS (Slide 20) adalah bantahan empirisnya; bias masuk lewat data, bukan lewat rumus.
5. **"Deep learning membuat ML klasik usang."** Untuk data tabular berukuran kecil-menengah, ensemble klasik (P5) masih sering menang; DL unggul di persepsi (citra, suara, teks).

### Demo live 10 menit (Slide 22–23)

Buka Colab kosong, ketik pipeline Iris sel per sel sambil narasi T/E/P: (1) load data + tampilkan `iris.frame.head()`; (2) split 70/30 dengan stratify; (3) fit LogisticRegression; (4) cetak classification_report dan baca precision/recall satu kelas bersama mahasiswa; (5) puncak demo — `model.predict([[9.9, 9.9, 9.9, 9.9]])`: model menjawab percaya diri untuk input mustahil. Tutup dengan kalimat: "model tidak tahu bahwa ia tidak tahu — tugas kalian sebagai engineer adalah tahu kapan model tidak boleh dipercaya." Siapkan notebook cadangan yang sudah jadi untuk antisipasi masalah koneksi.

### Pertanyaan diskusi (pilih 2, ±15 menit)

1. Sistem deteksi penyakit dengan akurasi 94% vs dokter 91%: apakah rumah sakit wajib memakainya? Siapa bertanggung jawab saat sistem salah? (menyambung Slide 18)
2. Jika dua definisi fairness pada kasus COMPAS terbukti tak bisa dipenuhi bersamaan, siapa yang berhak memilih definisi yang dipakai — engineer, pengadilan, atau publik? (Slide 20)
3. Klasifikasikan ke paradigma yang tepat dan definisikan T/E/P-nya: (a) mengelompokkan berita otomatis tanpa kategori awal; (b) mobil belajar parkir dari mencoba-coba; (c) prediksi mahasiswa berisiko DO dari data akademik.
4. "Data historis penerimaan kerja kita bias, jadi jangan pakai ML." Setuju atau tidak? Adakah jalan tengah antara menolak ML dan mereplikasi bias?

### Tips logistik

- Cek di awal kelas: semua mahasiswa bisa membuka Google Colab (butuh akun Google) — penentu kelancaran Tugas A.
- Bagikan link notebook demo (view-only) segera setelah kelas agar mahasiswa fokus mendengar, bukan menyalin.
- Kuis 5 menit di awal Pertemuan 2 tentang T/E/P dan tiga paradigma efektif memaksa pembacaan ulang materi ini.
