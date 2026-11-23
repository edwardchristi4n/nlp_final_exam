# Rancangan Pemodelan UAS — KelpNLP

**Topik:** Jenis Serangan Verbal dan Sasarannya pada Komentar Berita Polemik KPAI vs PB Djarum
**Acuan:** Spesifikasi Proyek UTS–UAS PBA 2026/2027 (Final v2), Bagian 6 (Proyek UAS) dan Bagian 7 (Rubrik UAS)
**Anggota:** Edward Christian Imanuel Tambunan (240712822) · Samuel Viva Gracia (240712828) · Rhiquelshen Charolim (240712846)
**Status:** rancangan, disusun 20 November 2026; belum ada kode yang dijalankan di folder proyek.

---

## 0. Ringkasan keputusan

| Aspek | Keputusan |
|---|---|
| Sumber data | `dataset/gold/KelpNLP_dataset_annotation.jsonl` (999 teks, 2.205 entitas, SHA-256 `8a8a77fa…6138b3d`) |
| Split | Satu split berbasis ID untuk **kedua** tugas, 70/15/15, *iterative stratification*, hampir-duplikat dikelompokkan, seed 42 |
| Multilabel — target | 5 label serangan (INSULTING, MOCKING, ACCUSING, PROVOKING, CURSING); NONE diturunkan bila tidak ada label terdeteksi |
| Multilabel — pembanding | Representasi: TF-IDF (n-gram kata + karakter) vs IndoBERT *fine-tuning*. Strategi: Binary Relevance vs Classifier Chain |
| NER — target | BIO, 9 tag (O + B-/I- untuk GOV_ORG, NONGOV_ORG, PERSON, GROUP) |
| NER — pembanding | Baseline CRF vs IndoBERT *token classification* |
| Pemilihan model terbaik | Berdasarkan skor validation; test set dipakai sekali untuk evaluasi akhir |
| Lingkungan | Lokal, laptop RTX 4050 (6 GB VRAM), fp16; Colab hanya cadangan |

Prinsip umum: taxonomy UTS tidak diubah (spesifikasi 6.1), semua langkah dapat dijalankan ulang dari file eksternal, dan setiap keputusan dijustifikasi dari data.

---

## 1. Karakteristik data yang menentukan desain

Diambil dari analisis gold dataset (20 Nov 2026).

| Temuan | Angka | Implikasi |
|---|---|---|
| Validitas span | 0 span dengan offset tidak cocok | Gold siap dipakai |
| Frekuensi label | ACCUSING 370, INSULTING 364, MOCKING 332, NONE 181, PROVOKING 168, **CURSING 7** | Imbalance sedang; CURSING praktis tidak dapat dipelajari |
| Label per teks | 1 label 646, 2 label 286, 3 label 64, 4 label 3 (multilabel 35,3%) | Masalah multilabel nyata |
| Kombinasi label | 21 kombinasi, 5 di antaranya muncul < 3 kali | Label Powerset tidak cocok |
| Korelasi antarlabel | \|φ\| ≤ 0,18 | Classifier Chain diperkirakan tidak banyak membantu (diuji) |
| Panjang teks | median 17 kata, p95 62, p99 122, maks 486 | `max_length` 256 subword (dicek ulang dengan tokenizer) |
| Token Prodigy (`tokens`, `ner_bio`) | 443 span menempel dengan tanda baca (mis. `KPAI,`) | `ner_bio` bawaan **tidak dipakai**; BIO dibangun ulang dari offset karakter |
| Dominasi O | ±91% token | Evaluasi NER wajib di tingkat entitas |
| GOV_ORG | 89% berupa "KPAI" | Mudah dihafal model |
| PERSON | 67 entitas; 31 dari 43 bentuk muncul sekali | Tipe entitas tersulit |
| Hampir-duplikat | ANC-2265≈2266, 3003≈3011, 2563≈3025, 1276≈1701, 1696≈1974 | Harus berada di split yang sama |
| Batas atas manusia | Fleiss κ rata-rata 0,840; NER F1 *exact* 0,926 | Acuan interpretasi skor model |

**Baseline cepat (5-fold CV, tanpa tuning) sebagai gambaran awal:**

- Multilabel TF-IDF kata+char + LR (BR): micro-F1 0,659; macro-F1 4 label 0,686. Per label: INSULTING 0,70 · MOCKING 0,54 · ACCUSING 0,64 · PROVOKING 0,86 · CURSING 0,00. N-gram karakter lebih baik daripada kata (+0,04 macro). Classifier Chain setara BR; Label Powerset lebih rendah (micro 0,590).
- NER CRF: micro-F1 0,919; GOV 0,974 · GROUP 0,891 · NONGOV 0,916 · **PERSON 0,386**. Pencarian kamus entitas dari data train saja sudah mencapai 0,918, sehingga skor tinggi sebagian besar berasal dari hafalan bentuk entitas.

Angka ini hanya orientasi. Angka resmi diambil dari split tetap di notebook.

---

## 2. Data preparation bersama (dipakai kedua tugas)

### 2.1 Loading dan validasi (rubrik 7.1.1 dan 7.2.1)

1. Baca gold dari file eksternal dengan path relatif dari `ROOT`.
2. Cek SHA-256 terhadap nilai rilis UTS.
3. Validasi struktur: kolom `id`, `text`, `labels`, `spans`, `meta`; ID unik; tidak ada teks kosong atau *missing*.
4. Validasi label: hanya dari taxonomy; NONE tidak bercampur label lain; setiap teks punya ≥ 1 label.
5. Validasi span: `text[start:end] == span.text`; tidak ada span bertumpuk; label entitas hanya dari 4 tipe; hitung jumlah entitas per tipe.
6. Cek duplikat persis (setelah *lowercase* dan rapikan spasi) dan hampir-duplikat (cosine TF-IDF karakter ≥ 0,75, lalu diperiksa manual).
7. Tampilkan contoh data dan statistik dasar sebagai bukti output.

Setiap pemeriksaan ditulis sebagai `assert` dengan pesan yang jelas, sehingga notebook berhenti bila data tidak sesuai.

### 2.2 Preprocessing teks

- Teks dipakai apa adanya dari gold (`text`, sudah melalui cleaning C1–C9b di UTS). Tidak ada normalisasi tambahan.
- Alasan: (a) offset span NER dihitung dari teks ini, sehingga mengubah teks akan merusak offset; (b) n-gram karakter dan tokenizer subword cukup tahan terhadap ejaan informal; (c) aplikasi cukup memakai satu fungsi preprocessing yang sama persis.
- Untuk TF-IDF: *lowercase* saja. Untuk IndoBERT: mengikuti tokenizer model (uncased).

### 2.3 Split train/validation/test (rubrik 7.1.2 dan 7.2.2)

| Pilihan | Keputusan dan alasan |
|---|---|
| Proporsi | 70/15/15 (±699/150/150), sesuai saran spesifikasi 6.2.1 |
| Metode | *Iterative stratification* (`iterstrat`), dua tahap: 70 vs 30, lalu 30 → 15/15 |
| Kunci stratifikasi | 5 label serangan + NONE + utas berita (1/2/3) + ada/tidaknya PERSON |
| Hampir-duplikat | Satu kelompok dianggap satu unit; seluruh anggota masuk split yang sama |
| Satu split untuk dua tugas | Spesifikasi 6.4 meminta perbandingan kedua model pada teks yang sama, dan teks itu harus belum pernah dilihat kedua model |
| CURSING | Diterima sebaran ±5/1/1; keterbatasan ini ditulis eksplisit |
| Seed | 42, dicatat |

Verifikasi setelah split:
- Tidak ada ID maupun teks (setelah normalisasi spasi/kapital) yang sama antar-split.
- Tabel proporsi setiap label, utas, dan tipe entitas per split dibandingkan dengan keseluruhan data.

Output (spesifikasi 6.6):
- `dataset/KelpNLP_multilabel_{train,val,test}.csv` — kolom: `id`, `text`, satu kolom biner per label (5 serangan + NONE), `labels` (string).
- `dataset/KelpNLP_ner_{train,val,test}.jsonl` — per baris: `id`, `text`, `spans` (start, end, text, label), `tokens` (teks, start, end), `tags` (BIO).

---

## 3. Proyek A — Multilabel Text Classification (50%)

### 3.1 Label engineering

- **Encoding:** `MultiLabelBinarizer` dengan urutan label tetap `[INSULTING, MOCKING, ACCUSING, PROVOKING, CURSING]`, disimpan bersama model.
- **NONE:** tidak dimodelkan sebagai output terpisah. NONE = tidak ada label serangan yang melewati threshold. Ini menjamin aturan guideline "NONE tidak boleh bersama label lain" selalu terpenuhi. NONE tetap dilaporkan sebagai baris di classification report.
- **CURSING:** tetap menjadi output model (taxonomy tidak diubah). Metriknya dilaporkan terpisah, dan macro-F1 disajikan dua versi: 5 label dan 4 label tanpa CURSING.

Alternatif yang dipertimbangkan untuk CURSING:

| Opsi | Alasan tidak dipilih |
|---|---|
| Dikeluarkan dari target | Mengubah taxonomy; melanggar 6.1 tanpa persetujuan dosen |
| Digabung ke INSULTING/PROVOKING | Mengubah taxonomy dan mengaburkan definisi label |
| Fitur kamus kata kutukan | Berisiko dinilai sebagai komponen *rule-based* |

### 3.2 EDA multilabel (rubrik 7.1.3)

Dikerjakan di notebook modeling (bukan hanya merujuk EDA UTS):
1. Panjang teks dalam kata dan dalam token subword IndoBERT; persentase teks terpotong pada `max_length` 128 dan 256.
2. Frekuensi tiap label, keseluruhan dan per split.
3. Jumlah label per teks (*label cardinality*, *label density*).
4. Kombinasi label dan matriks co-occurrence.
5. Matriks korelasi φ antarlabel.
6. Imbalance: rasio positif:negatif per label, IRLbl, MeanIR.
7. Panjang teks menurut jumlah label.

Setiap temuan ditutup dengan implikasinya, misalnya korelasi rendah → BR wajar, panjang teks → `max_length`, imbalance → `pos_weight` dan threshold per label.

### 3.3 Representasi teks (rubrik 7.1.4)

| Kode | Representasi | Parameter yang dicatat | Alasan |
|---|---|---|---|
| R1 | TF-IDF n-gram kata (1–2) + n-gram karakter `char_wb` (2–5), digabung | `min_df`, `sublinear_tf`, ukuran vocabulary | Cepat, interpretable; n-gram karakter tahan ejaan informal (`tolooool`, `anak2`, `bubarkanKPAI`) |
| R2 | IndoBERT *fine-tuning* (embedding kontekstual) | nama model, `max_length`, tokenizer | Konteks kalimat penting untuk MOCKING (sarkasme) yang sulit bagi n-gram |

- **Pencegahan leakage:** vectorizer TF-IDF di-*fit* hanya pada train, lalu `transform` untuk val/test. Semua dibungkus dalam `Pipeline`.
- **Kandidat model R2:** `indolem/indobertweet-base-uncased` (dilatih dari tweet Indonesia, gaya bahasa mirip komentar) dan `indobenchmark/indobert-base-p1`. Keduanya dibandingkan di validation; yang lebih baik dipakai sebagai R2 final.
- Sebagai ablasi kecil di dalam R1: kata saja vs karakter saja vs gabungan, supaya pilihan gabungan terjustifikasi.

### 3.4 Model dan strategi multilabel (rubrik 7.1.5)

| Eksperimen | Representasi | Algoritma | Strategi | Penanganan imbalance |
|---|---|---|---|---|
| M1 | R1 | Logistic Regression | Binary Relevance | `class_weight="balanced"` |
| M2 | R1 | Logistic Regression | Classifier Chain (urutan dari label paling sering) | `class_weight="balanced"` |
| M3 | R1 | Linear SVM + kalibrasi probabilitas | Binary Relevance | `class_weight="balanced"` |
| M4 | R2 | IndoBERT + kepala sigmoid 5 output | Binary Relevance (satu model, 5 sigmoid) | BCE dengan `pos_weight` = negatif/positif per label |

Label Powerset tidak dijadikan eksperimen utama karena 21 kombinasi dengan banyak kombinasi langka; alasan ini ditulis dengan merujuk tabel kombinasi di EDA.

**Tuning (hanya memakai validation set):**
- M1–M3: grid kecil `C ∈ {0,25; 1; 4; 16}`, dipilih berdasarkan macro-F1 4 label di validation.
- M4: learning rate `{2e-5, 3e-5, 5e-5}`, maks 10 epoch, batch 8 × *gradient accumulation* 2, fp16, *weight decay* 0,01, *warmup* 10%, **early stopping** (patience 2) dengan metrik macro-F1 validation; checkpoint terbaik disimpan.

**Pencatatan eksperimen:** satu DataFrame berisi setiap konfigurasi (representasi, model, strategi, hyperparameter, seed, skor validation), disimpan ke `models/KelpNLP_multilabel_experiments.csv`.

### 3.5 Threshold (decision rule)

- Default 0,5 dibandingkan dengan **threshold per label** yang dicari di validation (grid 0,05–0,95, memaksimalkan F1 per label).
- CURSING memakai threshold default karena contoh validation hanya ±1, sehingga threshold hasil tuning tidak bermakna.
- Threshold final disimpan bersama model dan ditampilkan di aplikasi.

### 3.6 Evaluasi (rubrik 7.1.6)

Semua model final diuji pada **test set yang sama, satu kali**:
- Classification report per label (precision, recall, F1, support), termasuk NONE turunan.
- Micro-F1, macro-F1 (5 dan 4 label), weighted-F1, Hamming loss, subset accuracy.
- Tabel perbandingan M1–M4, plus perbandingan threshold 0,5 vs threshold hasil tuning.
- Confusion matrix per label (`multilabel_confusion_matrix`).

### 3.7 Error analysis

1. **False positive dan false negative per label**, masing-masing dengan contoh teks, label emas, prediksi, dan probabilitas.
2. **Pasangan label yang sering tertukar**, terutama MOCKING vs INSULTING (literal vs tidak literal sesuai guideline 3.2).
3. **Label langka:** perilaku model pada CURSING.
4. **Teks multilabel:** apakah model cenderung hanya memprediksi satu label (*under-prediction*).
5. **Pengaruh panjang teks:** skor pada teks pendek vs panjang (dan teks yang terpotong).
6. **Perbandingan model:** contoh yang benar di IndoBERT tetapi salah di TF-IDF, dan sebaliknya.

Prediksi test disimpan ke `models/KelpNLP_multilabel_test_predictions.csv`.

### 3.8 Model yang disimpan

- `models/KelpNLP_best_multilabel_model/` — bobot model, tokenizer/vectorizer, urutan label, threshold, dan `config` preprocessing, dalam format library yang dipakai (`save_pretrained` atau `joblib`).

---

## 4. Proyek B — Named Entity Recognition (50%)

### 4.1 Tokenisasi dan konversi tag (rubrik 7.2.2)

**Masalah:** `tokens`/`ner_bio` di gold memakai token Prodigy yang dipisah per spasi, sehingga tanda baca ikut menempel (mis. `KPAI,` → B-GOV_ORG). Bila dipakai langsung, span hasil prediksi membawa tanda baca, dihitung sebagai *boundary error*, dan highlight di aplikasi tidak rapi.

**Solusi:**
1. Tokenisasi ulang dengan regex `\w+(?:[-']\w+)*|[^\w\s]`: kata (termasuk `anak-anak`) dan setiap tanda baca menjadi token terpisah, masing-masing dengan offset karakter.
2. BIO dibangun dari offset karakter span gold: token pertama di dalam span → `B-`, token berikutnya → `I-`, sisanya `O`.
3. **Uji balik:** span → BIO → span harus identik untuk seluruh 2.205 entitas (sudah dicoba: 0 span tidak pas dengan batas token baru).
4. Ditampilkan beberapa contoh token–tag sebagai bukti, termasuk kasus `KPAI,`, `PB. Djarum`, dan entitas dalam tanda kutip.

**Skema tag:** BIO. BIOES tidak dipakai karena tag I- sudah sedikit (I-PERSON 43, I-GOV_ORG 37) dan BIOES akan memecahnya lagi menjadi kelas yang lebih kecil.

**Alignment subword (khusus IndoBERT):**
- Input berupa daftar kata (`is_split_into_words=True`), memakai `word_ids()`.
- Label diberikan ke subword pertama setiap kata; subword lanjutan, `[CLS]`, `[SEP]`, dan padding diberi `-100` (diabaikan loss).
- Verifikasi: cetak tabel subword → word_id → label untuk beberapa contoh, dan uji otomatis bahwa jumlah label non-(-100) sama dengan jumlah kata yang tidak terpotong.
- **Truncation:** `max_length` 256. Teks yang lebih panjang (maks ±600 token regex) diproses dengan *sliding window* (stride 64) saat inferensi, sehingga tidak ada entitas yang hilang. Jumlah teks yang terkena dicatat.

### 4.2 EDA entitas (rubrik 7.2.3)

1. Jumlah token per teks (token regex dan subword).
2. Jumlah entitas per teks (0, 1, 2, …).
3. Distribusi entitas per tipe, keseluruhan dan per split.
4. Panjang span dalam token per tipe.
5. Frekuensi O vs tag entitas (B-/I- per tipe), rasio O:entitas.
6. Keragaman bentuk entitas per tipe (bentuk unik, bentuk yang muncul sekali), dengan GOV_ORG ("KPAI" 89%) vs PERSON (paling beragam).
7. Contoh entitas representatif dan ambigu, mis. "atlit" (GROUP/NONGOV_ORG), "pemerintah", entitas di dalam tanda kutip.

Implikasi yang ditarik: evaluasi tingkat entitas, perhatian khusus pada PERSON, stratifikasi PERSON pada split.

### 4.3 Representasi dan tag mapping (rubrik 7.2.4)

- `label2id`/`id2label` untuk 9 tag dengan urutan tetap, disimpan di config model, dan diuji konsisten dua arah.
- **CRF:** fitur per token, yaitu bentuk kecil, prefiks/sufiks 3 huruf, kapitalisasi, angka, panjang, serta kata dan kapitalisasi di jendela ±2.
- **IndoBERT:** embedding subword dari model pretrained; batching dengan `DataCollatorForTokenClassification` (padding dinamis, attention mask, label padding `-100`).
- **Dominasi O:** dibandingkan dua konfigurasi, yaitu tanpa bobot kelas vs bobot kelas ringan untuk tag entitas. Dipilih berdasarkan F1 entitas di validation, dengan memperhatikan recall PERSON.

### 4.4 Model dan training (rubrik 7.2.5)

| Eksperimen | Model | Peran |
|---|---|---|
| N1 | CRF (`sklearn-crfsuite`, L-BFGS) | Baseline machine learning |
| N2 | IndoBERT *token classification* | Model kuat (advanced) |

**Tuning (hanya memakai validation set):**
- N1: grid `c1, c2 ∈ {0,01; 0,1; 0,5}`, maks 200 iterasi.
- N2: model kandidat sama dengan multilabel (IndoBERTweet vs IndoBERT-p1); learning rate `{3e-5, 5e-5}`, maks 15 epoch, batch 8 × accumulation 2, fp16, early stopping (patience 3) dengan metrik F1 entitas validation, checkpoint terbaik disimpan.
- Seed, hyperparameter, dan *training history* (loss train/val dan F1 val per epoch) dicatat dan diplot.

**Pencatatan eksperimen:** `models/KelpNLP_NER_experiments.csv`.

### 4.5 Evaluasi (rubrik 7.2.6)

Pada **test set yang sama, satu kali**, untuk N1 dan N2:
- `seqeval` mode *strict* skema IOB2: precision, recall, F1 tingkat entitas, per tipe dan micro/macro.
- Token-level metric hanya sebagai tambahan, tidak menggantikan metrik entitas.
- Prediksi disimpan sebagai span (start, end, label) ke `models/KelpNLP_NER_test_predictions.jsonl`, supaya dapat diperiksa.
- Ditambahkan satu kalimat interpretasi dengan acuan F1 kesepakatan manusia 0,926.

### 4.6 Error analysis NER

Setiap pasangan span emas–prediksi dikelompokkan otomatis:

| Kategori | Definisi |
|---|---|
| Benar | Batas dan tipe sama persis |
| Boundary error | Tumpang tindih, tipe sama, batas berbeda (mis. `PB Djarum` vs `Djarum`) |
| Label error | Batas sama, tipe berbeda (mis. GROUP vs NONGOV_ORG) |
| Boundary + label | Tumpang tindih, batas dan tipe berbeda |
| Missed entity (FN) | Span emas tanpa prediksi yang tumpang tindih |
| False positive (spurious) | Prediksi tanpa span emas yang tumpang tindih |

Untuk setiap kategori: jumlah per tipe entitas dan contoh nyata. Pembahasan tambahan:
- **Entitas yang tidak muncul di train vs yang sudah dikenal:** recall keduanya dibandingkan, untuk memisahkan kemampuan generalisasi dari hafalan (khususnya GOV_ORG "KPAI" dan PERSON).
- **Kasus ambigu** sesuai guideline: "atlit", "pemerintah", singkatan vs nama panjang, entitas dalam tanda kutip.
- Perbandingan N1 vs N2 pada contoh yang sama.

### 4.7 Model yang disimpan

- `models/KelpNLP_best_NER_model/` — bobot, tokenizer, `label2id`/`id2label`, dan aturan tokenisasi regex (atau `.joblib` untuk CRF bila CRF yang terbaik).

---

## 5. Analisis hubungan kedua model (spesifikasi 6.4)

**Catatan desain:** NER proyek ini menandai **pihak yang disebut**, bukan sasaran serangan dan bukan aspek bersentimen seperti contoh di spesifikasi. Karena itu konsistensi didefinisikan sebagai berikut.

| Kategori | Kondisi (pada prediksi) | Contoh pola |
|---|---|---|
| Konsisten | Ada label serangan dan ada entitas | PROVOKING + GOV_ORG ("Bubarkan KPAI") |
| Konsisten | NONE dan ada entitas | Menyebut pihak tanpa menyerang (12–26% teks berentitas di gold berlabel NONE) |
| Sebagian konsisten | Ada label serangan tanpa entitas | Sasaran implisit ("dia", "mereka") |
| Kontradiktif / perlu dicek | Pola yang jarang di gold, atau salah satu model keliru terhadap gold | Dibahas penyebabnya |

Langkah: jalankan kedua model terbaik pada seluruh test set → hitung distribusi kategori → bandingkan dengan distribusi pada label emas → bahas 5–8 contoh beserta kemungkinan penyebabnya (kesalahan model mana, ambiguitas teks, atau keterbatasan definisi).

---

## 6. Deployment (ringkas, spesifikasi 6.5)

`streamlit/KelpNLP_NLP_app.py` memuat model terbaik yang tersimpan, memakai preprocessing yang sama dengan training:
- **Prediksi Multilabel:** input teks, label terprediksi, probabilitas per label, threshold, interpretasi singkat (termasuk catatan CURSING sebagai label langka).
- **Prediksi NER:** input teks, daftar entitas dan label, highlight span pada teks (offset dari teks input asli), interpretasi singkat.
- **Integrated Analysis (dianjurkan):** kedua prediksi pada teks yang sama, beserta kategori konsistensi dari bagian 5.

Model IndoBERT (±500 MB per model) di-host di Hugging Face Hub agar repositori GitHub dan Streamlit Cloud dapat memakainya.

---

## 7. Lingkungan dan reproducibility

| Aspek | Ketentuan |
|---|---|
| Perangkat | Laptop RTX 4050 6 GB; fp16; batch 8 × accumulation 2; `gradient_checkpointing` bila OOM |
| Path | Satu variabel `ROOT` (deteksi lokal/Colab); semua path relatif |
| Library | Versi dikunci di `requirements.txt` dan dicetak di awal notebook; PyTorch versi CUDA (`torch.cuda.is_available()` harus `True`) |
| Windows | `dataloader_num_workers=0` |
| Seed | 42 untuk `random`, `numpy`, `torch`, `transformers.set_seed` |
| Penilai tanpa GPU | Flag `TRAIN_TRANSFORMER = False` memuat model tersimpan lalu langsung evaluasi; output training asli tetap tersimpan di notebook |

---

## 8. Output dan pemetaan ke rubrik

| File (spesifikasi 6.6) | Isi |
|---|---|
| `dataset/KelpNLP_multilabel_{train,val,test}.csv` | Split multilabel |
| `dataset/KelpNLP_ner_{train,val,test}.jsonl` | Split NER (span + token + BIO) |
| `notebook/KelpNLP_multilabel_modeling.ipynb` | Bagian 2–3 |
| `notebook/KelpNLP_NER_modeling.ipynb` | Bagian 2 dan 4 (split dibaca dari file, tidak dibuat ulang), lalu bagian 5 |
| `models/KelpNLP_best_multilabel_model/`, `models/KelpNLP_best_NER_model/` | Model terbaik |
| `streamlit/KelpNLP_NLP_app.py` | Bagian 6 |
| `report/KelpNLP_laporan_UAS.pdf`, `code_pdf/KelpNLP_code_export_UAS.pdf`, `KelpNLP_anggota.txt` | Laporan, export kode, anggota + link repo, deployment, video |

| Rubrik | Bagian rancangan |
|---|---|
| 7.1.1 Data Loading & Validation (10%) | 2.1 |
| 7.1.2 Preprocessing & Label Engineering (15%) | 2.2, 2.3, 3.1 |
| 7.1.3 EDA Multilabel (10%) | 3.2 |
| 7.1.4 Text Representation (15%) | 3.3 |
| 7.1.5 Model Training & Experiment (20%) | 3.4, 3.5 |
| 7.1.6 Evaluation & Error Analysis (20%) | 3.6, 3.7 |
| 7.1.7 Deployment Multilabel (10%) | 3.8, 6 |
| 7.2.1 Data Loading & Span Validation (10%) | 2.1 |
| 7.2.2 Tokenization & Tag Conversion (15%) | 2.3, 4.1 |
| 7.2.3 EDA Entity (10%) | 4.2 |
| 7.2.4 Embedding & Tag Mapping (15%) | 4.3 |
| 7.2.5 Model Configuration & Training (20%) | 4.4 |
| 7.2.6 Evaluation & Error Analysis (20%) | 4.5, 4.6 |
| 7.2.7 Deployment NER (10%) | 4.7, 6 |

---

## 9. Risiko dan keterbatasan yang dilaporkan

1. **CURSING (7 teks):** metrik tidak stabil, dilaporkan terpisah.
2. **Ukuran test (±150 teks):** F1 per label dan PERSON (±10 entitas di test) bervariasi; jumlah support selalu ditampilkan.
3. **Hafalan entitas:** GOV_ORG didominasi "KPAI"; kemampuan model dinilai juga dari entitas yang belum pernah dilihat.
4. **Distribusi label sampel:** gold berasal dari sampel yang diperkaya teks berindikasi serangan (47,6% strata prioritas), sehingga proporsi label bukan proporsi seluruh komentar.
5. **Definisi NER:** entitas = pihak yang disebut, bukan sasaran; memengaruhi interpretasi analisis 6.4.
