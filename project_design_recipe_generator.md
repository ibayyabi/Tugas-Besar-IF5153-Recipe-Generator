# Team Project - Recipe Generator

**Diajukan untuk Memenuhi Tugas Besar Mata Kuliah Pemrosesan Bahasa Natural Lanjut pada Semester Ganjil Tahun Akademik 2026-2027**

**Oleh:**  
**Team 13**

| Nama | NIM |
| :--- | :--- |
| Mudzaki Kaarzaqiel Hakim | 18223024 |
| Mochamad Ikhbar A | 18223050 |
| Raditya Zaki Athaya | 18223086 |

**SEKOLAH TEKNIK ELEKTRO DAN INFORMATIKA**  
**INSTITUT TEKNOLOGI BANDUNG**  
**2026**

---

## 1. PROJECT DESIGN

### 1.1. Deskripsi Projek

**Recipe Generator** adalah sebuah sistem rekomendasi resep berbasis NLP yang dapat membantu pengguna menemukan resep berdasarkan:
* Bahan makanan yang saat ini dimiliki
* Bahan yang tidak ingin digunakan
* Target atau batas kalori
* Jumlah bahan tambahan yang masih perlu dibeli

Sistem ini dapat melakukan *multi-criteria recommendation* dengan mempertimbangkan:
* *Semantic/lexical ingredient similarity*
* *Ingredient coverage*
* Jumlah *missing ingredients*
* *Calorie compatibility*
* *Excluded ingredient violations*

**Contoh Input:**
> “I have chicken, potato, carrot, onion, and egg. I want something around 500 calories and no cheese”

**Contoh Output:**
```text
1. Chicken Potato Soup
Overall match : XX%
Ingredient coverage : XX%
Estimated calories : XXX kcal
Calorie match : XX%
Available:
- Chicken
- potato
- carrot
- onion
Missing:
- celery
Excluded ingredient:
- no cheese detected
```

Untuk sistem ini, kami menggunakan dataset global yang sudah ada (seperti RecipeNLG) yang dapat diakses melalui repositori GitHub dan publikasi paper terkait.

---

### 1.2. NLP Pipeline

#### 1.2.1. Master Data Preparation

Tahap *Master Data Preparation* merupakan fondasi utama sebelum sistem dapat menjalankan proses pencarian dan rekomendasi. Pada tahap ini, sistem menerima masukan data mentah dari dataset **RecipeNLG**, yang kemudian diproses pada komponen **Recipe Preprocessing & Enrichment**. Proses ini melibatkan pembersihan data, normalisasi satuan bahan dengan bantuan **Ingredient Alias Dictionary**, serta pengayaan data kalori yang merujuk pada **Calorie Reference Dataset**.

Seluruh proses *preprocessing* ini berjalan di belakang layar untuk menghasilkan **Processed Recipe Corpus**. Korpus yang sudah bersih dan kaya akan informasi gizi inilah yang selanjutnya diubah menjadi representasi matematis melalui tahap **TF-IDF Fitting** untuk menghasilkan **Recipe TF-IDF Matrix** agar siap digunakan oleh model *retrieval* di Pipeline 2.

**Konteks & Catatan Sistem (Diagram BEAM 1.1):**
* **Context:** Sistem digunakan oleh pengguna yang ingin mengidentifikasi resep yang dapat dimasak menggunakan bahan yang tersedia di dapur (*pantry*) sambil meminimalkan bahan tambahan serta memenuhi batasan kalori dan bahan secara aproksimatif.
* **Note:** Nilai kalori resep merupakan estimasi yang diturunkan dari kuantitas bahan dan data referensi kalori, bukan merupakan informasi nutrisi yang otoritatif.

**Analisis Risiko & Mitigasi (BEAM):**
* **Risk Source:** *Incomplete Quantity and Unit Information* (Bahan resep dapat mengandung informasi kuantitas dan satuan yang tidak lengkap, ambigu, atau sulit dinormalisasi).
* **Risk:** *Inaccurate Calorie Estimation* (Estimasi kalori resep berbeda secara substansial dari kandungan kalori sebenarnya).
* **Consequence:** *Incorrect Calorie Compatibility Score* (Resep menerima skor kompatibilitas kalori yang tidak tepat saat proses *reranking*).
* **Impact:** *Calorie Target Not Satisfied* (Pengguna dapat memilih resep yang tidak mendekati target kalori yang diinginkan).
* **Risk Control:** *Calorie Mapping Validation* (Memvalidasi *parsing* kuantitas, konversi satuan, dan pencocokan referensi kalori; serta menampilkan nilai kalori secara eksplisit sebagai estimasi).

---

#### 1.2.2. Pipeline 1 - Query & Constraint Understanding (18223050)

Pipeline pertama berfungsi untuk mengubah masukan pengguna dalam bentuk bahasa alami menjadi representasi terstruktur yang dapat diproses oleh sistem pada tahap berikutnya. Masukan awal berupa *natural language query* akan diproses melalui **Query & Constraint Parsing** (*Transformation*). Proses transformasi ini mencakup normalisasi teks, ekstraksi bahan makanan, normalisasi nama bahan, identifikasi bahan yang dikecualikan, serta ekstraksi batas atau target kalori. Seluruh proses dilakukan menggunakan pendekatan *rule-based*.

Dalam notasi BEAM, **Ingredient Vocabulary** dan **Constraint Patterns** direpresentasikan sebagai *Symbol* karena keduanya menjadi pengetahuan eksplisit yang digunakan sistem untuk mengenali bahan dan pola *constraint*. Hasil akhir pipeline adalah **Structured Query**, yang juga direpresentasikan sebagai *Symbol* karena informasi dari teks bebas telah diubah menjadi struktur eksplisit berupa *available ingredients*, *excluded ingredients*, dan *calorie target*. Query baru ini akan menjadi input untuk Pipeline 2 dan Pipeline 3.

Pipeline ini memiliki peran penting karena kesalahan interpretasi pada tahap awal dapat memengaruhi seluruh proses rekomendasi berikutnya. 

**Analisis Risiko & Mitigasi (BEAM):**
1. **Alur Risiko Ambiguitas Ekspresi:**
   * **Risk Source:** *Ambiguous Ingredient Expressions* (Kueri pengguna dapat mengandung bentuk jamak/plural, sinonim, nama bahan informal, atau ekspresi makanan majemuk).
   * **Risk:** *Incorrect Ingredient Extraction* (Bahan yang tersedia atau dikecualikan salah diidentifikasi dari kueri pengguna).
   * **Consequence:** *Incorrect Structured Query* (Batasan *pantry* atau pengecualian yang terstruktur tidak lagi merepresentasikan permintaan asli pengguna).
   * **Impact:** *Irrelevant Recipe Recommendation* (Pengguna menerima rekomendasi resep yang tidak sesuai dengan bahan dapur atau preferensi mereka).
   * **Risk Control:** *Ingredient Normalization & Alias Mapping* (Menerapkan lemmatisasi, pemetaan bahan kanonik, kamus alias, dan aturan pengecualian manual / *exception rules*).
2. **Alur Risiko Ketidakcocokan Kosakata:**
   * **Risk Source:** *Ingredient Vocabulary Mismatch* (Bahan yang ekuivalen direpresentasikan menggunakan bentuk permukaan kata yang berbeda).
   * **Risk:** *Incorrect Ingredient Matching* (Bahan yang secara semantik ekuivalen diperlakukan sebagai bahan yang berbeda).
   * **Consequence:** *Incorrect Pantry Coverage* (Bahan yang sebenarnya tersedia keliru diklasifikasikan sebagai hilang, sehingga menurunkan skor cakupan bahan).
   * **Impact:** *Unnecessary Ingredient Purchases* (Pengguna diberi tahu bahwa suatu bahan tidak tersedia padahal bahan ekuivalennya sudah dimiliki).

---

#### 1.2.3. Pipeline 2 - Candidate Recipe Retrieval (18223024)

**Pipeline 2 (Candidate Recipe Retrieval)** berfungsi untuk mengambil kumpulan kandidat resep yang secara tekstual atau komposisi relevan dengan bahan makanan pengguna. Tahapan ini memiliki fokus utama pada pengoptimalan kecepatan pencarian (*retrieval speed*) dan pencapaian *recall* yang tinggi. Pendekatan ini digunakan karena melakukan perhitungan *multi-criteria scoring* secara langsung terhadap ratusan ribu hingga jutaan resep pada korpus utama sangatlah tidak efisien. Oleh karena itu, sistem menggunakan pendekatan pencarian dua tahap (*two-stage retrieval*) di mana Pipeline 2 bertugas menyaring data besar menjadi himpunan kandidat kecil (misalnya Top-$N$ atau Top-20 resep) sebelum dilanjutkan ke tahap *reranking*.

Proses operasional pada pipeline ini dimulai dengan menerima masukan berupa kueri terstruktur (*Structured Query*) yang telah dihasilkan oleh Pipeline 1. Kueri tersebut kemudian diproses melalui tahap **Query Vectorization** dengan mengandalkan metode **TF-IDF** (*Term Frequency-Inverse Document Frequency*). Proses ini bertujuan untuk mengubah representasi teks dari bahan-bahan yang dimiliki pengguna menjadi sebuah representasi matematis dalam bentuk vektor kueri (**Query Vector**).

Selanjutnya, sistem akan membandingkan vektor kueri pengguna tersebut dengan kumpulan vektor matriks resep (**Recipe TF-IDF Matrix**). Proses pencocokan ini dijalankan pada komponen **Similarity Retrieval Inference** yang menggunakan perhitungan *cosine similarity* untuk mengukur jarak kemiripan antar vektor. Dalam notasi BEAM:
* *Transform:* **Query Vectorization**
* *Data:* **Query Vector**, **Recipe TF-IDF Matrix**, dan **Candidate Recipes**
* *Deduce:* **Similarity Retrieval Inference** (mengubah aliran data matriks menjadi sekumpulan kandidat resep sebagai keluaran akhir pipeline).

**Analisis Risiko & Mitigasi (BEAM):**
* **Risk Source:** *Frequent Pantry Staples* (Bahan-bahan dapur generik seperti garam, air, minyak, dan lada memiliki frekuensi kemunculan yang sangat tinggi di berbagai resep).
* **Risk:** *Biased Similarity Retrieval* (Proses pengambilan resep kandidat didominasi oleh kecocokan bahan-bahan generik yang sering muncul alih-alih bahan utama resep yang lebih bermakna).
* **Consequence:** *Low-Quality Candidate Set* (Resep-resep yang sebenarnya relevan menjadi tidak terwakili atau *underrepresented* dalam daftar kandidat Top-$N$ yang diteruskan ke tahap *reranking*).
* **Impact:** *Reduced Recommendation Relevance* (Resep yang relevan gagal muncul dalam daftar rekomendasi akhir Top-$K$ untuk pengguna).
* **Risk Control:** *Pantry Staple Down-Weighting Control* (Mengurangi bobot penilaian / *down-weight* pada bahan-bahan dapur generik selama tahapan pencarian dan penilaian rekomendasi agar relevansi tetap berpusat pada bahan-bahan utama).

---

#### 1.2.4. Pipeline 3 - Multi Criteria Recipe Reranking (18223086)

**Pipeline 3 (Multi-Criteria Recipe Reranking)** berfungsi untuk melakukan pengurutan kembali (*reranking*) terhadap daftar kandidat resep berdasarkan tingkat kebergunaan (*usability*) resep tersebut bagi pengguna. Tahapan ini merupakan diferensiasi utama dari sistem dibandingkan dengan sistem pencarian biasa. Pada tahap ini, rekomendasi tidak lagi sekadar dievaluasi berdasarkan kemiripan tekstual semata, melainkan dievaluasi secara komprehensif dengan menghitung:
* Persentase kecocokan bahan (*ingredient coverage*)
* Penalti untuk bahan yang tidak dimiliki (*missing ingredient penalty*)
* Kompatibilitas target kalori (*calorie compatibility*)
* Pemberlakuan filter untuk bahan yang dilarang (*excluded ingredient penalty*)

Proses operasional pada pipeline ini dimulai dengan menerima dua buah masukan sekaligus, yaitu **Structured Query** (keluaran dari Pipeline 1) dan **Candidate Recipes** (hasil pencarian dari Pipeline 2). Kedua komponen masukan ini kemudian diproses ke dalam komponen utama bernama **Multi-criteria Scoring & Filtering**. Dalam prosesnya, sistem juga mempertimbangkan bobot kepentingan dari bahan-bahan dapur dasar (seperti garam atau air) agar pengguna tidak mendapatkan penalti yang besar jika tidak menyebutkan bahan generik tersebut di awal.

Dalam notasi BEAM:
* *Deduce:* **Multi-criteria Scoring & Filtering** (melakukan perhitungan logis dan inferensi skor akhir untuk menyaring data kandidat).
* *Symbol:* **Ranking Configuration** dan **Pantry Staple Weights** (berisi aturan parameter matematika dan pembobotan eksplisit yang digunakan oleh sistem).
* *Data:* **Top-K Recipes** (keluaran akhir berupa daftar rekomendasi resep terbaik).

**Catatan Rancangan (BEAM):**
* Komponen ini tidak memiliki rantai risiko (*Risk*) secara langsung, melainkan dilengkapi dengan sebuah anotasi **Note** yang melekat pada *Ranking Configuration*: *"Initial ranking weights are experimental and must be validated through evaluation."*
* Bobot metrik awal yang digunakan untuk perhitungan skoring (seperti persentase untuk TF-IDF, *coverage* bahan, atau kalori) pada dasarnya masih bersifat eksperimental dan harus divalidasi kembali melalui tahap evaluasi (*ablation study*) di akhir proyek untuk memastikan sistem menghasilkan rekomendasi yang optimal dan relevan bagi pengguna.

---

#### 1.2.5. Full Pipeline

Secara keseluruhan, arsitektur **Recipe Generator** ini membentuk sebuah ekosistem *Natural Language Processing* (NLP) yang berjalan secara sekuensial (berurutan) tanpa bergantung pada *Large Language Model* (LLM) generatif:

1. **Master Data Preparation:** Membersihkan dan memperkaya korpus resep mentah (RecipeNLG) dengan referensi kalori dan kamus alias, lalu membangun matriks TF-IDF.
2. **Pipeline 1 (Query & Constraint Understanding):** Menerjemahkan bahasa alami pengguna menjadi kueri terstruktur (*available ingredients*, *excluded ingredients*, dan *calorie target*).
3. **Pipeline 2 (Candidate Recipe Retrieval):** Melakukan pencarian resep relevan dalam jumlah besar secara cepat berbasis kemiripan vektor (TF-IDF & *cosine similarity*) untuk menghasilkan himpunan kandidat resep.
4. **Pipeline 3 (Multi-Criteria Recipe Reranking):** Menyaring dan mengurutkan kembali daftar kandidat resep berdasarkan logika kecocokan bahan asli di dapur pengguna, penalti bahan kurang/terlarang, serta target kalori untuk menghasilkan rekomendasi akhir (*Top-K Recipes*).

Integrasi seluruh pipeline ini memastikan rekomendasi yang diberikan tidak hanya relevan secara teks, tetapi juga realistis untuk dimasak (*cookable*).