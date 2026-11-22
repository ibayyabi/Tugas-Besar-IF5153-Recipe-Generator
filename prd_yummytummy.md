# Product Requirements Document: Pantry-Aware & Calorie-Constrained Recipe Recommender

* **Document Type:** Product Requirements Document
* **Primary Dataset:** RecipeNLG
* **Project Domain:** Natural Language Processing / Information Retrieval
* **Project Type:** Academic NLP Group Project
* **Core Constraint:** No frontier LLM / external LLM API
* **Status:** Draft / Single Source of Truth

---

## 1. Product Overview

### 1.1 Product Name
* **Official Name:** Pantry-Aware & Calorie-Constrained Recipe Recommender
* **Working Name:** YummyTummy

### 1.2 Product Summary
**YummyTummy** adalah sistem rekomendasi resep berbasis NLP yang membantu pengguna menemukan resep berdasarkan:
1. Bahan makanan yang saat ini dimiliki;
2. Bahan yang tidak ingin digunakan;
3. Target atau batas kalori;
4. Jumlah bahan tambahan yang masih perlu dibeli.

Berbeda dari *recipe recommender* konvensional yang hanya mencari resep berdasarkan kemiripan teks, sistem ini melakukan *multi-criteria recommendation* dengan mempertimbangkan:
* *Semantic/lexical ingredient similarity*;
* *Ingredient coverage*;
* Jumlah *missing ingredients*;
* *Calorie compatibility*;
* *Excluded ingredient violations*.

**Contoh Input:**
> "I have chicken, potato, carrot, onion, and egg. I want something around 500 calories and no cheese."

**Contoh Output:**
```text
1. Chicken Potato Soup
Overall Match       : 91%
Ingredient Coverage : 86%
Estimated Calories  : 482 kcal
Calorie Match       : 96%

Available:
✓ chicken
✓ potato
✓ carrot
✓ onion

Missing:
✗ celery

Excluded Ingredient:
✓ no cheese detected
```

Sistem tidak menggunakan *Large Language Model* (LLM) sebagai komponen utama. NLP dilakukan menggunakan kombinasi:
* *Text normalization*;
* *Rule-based information extraction*;
* TF-IDF;
* *Cosine similarity*;
* *Ingredient set similarity*;
* *Heuristic / weighted reranking*.

---

## 2. Background

*Recipe recommendation* sering kali hanya menggunakan kemiripan antara kueri pengguna dengan kumpulan resep. Sebagai contoh, kueri `"chicken potato carrot onion"` dapat dibandingkan dengan daftar bahan setiap resep menggunakan:

$$\text{TF-IDF} + \text{Cosine Similarity}$$

Masalahnya, kemiripan tekstual saja belum cukup menjawab kebutuhan nyata pengguna. Dua resep dapat memiliki nilai *similarity* yang tinggi, tetapi:
* Salah satunya memerlukan banyak bahan tambahan;
* Salah satunya berada jauh dari target kalori pengguna;
* Salah satunya mengandung bahan yang secara eksplisit tidak diinginkan pengguna.

Karena itu, objektif **YummyTummy** bukan hanya:
> *"Find the recipe most textually similar to the user's ingredients."*

Melainkan:
> *"Find recipes that are practically cookable using the user's current pantry while satisfying explicit calorie and ingredient constraints."*

---

## 3. Reference Notebook and Differentiation

Notebook referensi menggunakan dataset **RecipeNLG** dan secara umum memiliki alur:
```text
User Input 
  ↓
Ingredient Extraction 
  ↓
Calorie Constraint Extraction 
  ↓
Emotion Detection 
  ↓
Recipe Filtering 
  ↓
TF-IDF + Cosine Similarity 
  ↓
Top-3 Recipes
```

Notebook tersebut juga telah memiliki *preprocessing* untuk dataset RecipeNLG:
* *Ingredient cleaning*;
* *Ingredient quantity parsing*;
* *Unit parsing*;
* *Calorie estimation*;
* *Fuzzy ingredient matching*;
* *TF-IDF vectorization*.

**YummyTummy** akan menggunakan sebagian besar data dan *preprocessing* tersebut, tetapi mengganti *recommendation objective*:

| Existing Objective | Proposed Objective |
| :--- | :--- |
| • Ingredient similarity<br>• Calorie filter<br>• Emotion | • Ingredient relevance<br>• Pantry coverage<br>• Missing ingredient penalty<br>• Excluded ingredient penalty<br>• Calorie compatibility |

*Catatan: Emotion classification tidak menjadi core feature karena hubungan antara mood dan kecocokan resep sulit untuk didefinisikan dan dievaluasi secara objektif.*

---

## 4. Problem Statement

Pengguna sering memiliki beberapa bahan makanan tetapi tidak mengetahui resep apa yang paling *feasible* untuk dibuat. *Traditional recipe search* dapat menghasilkan resep yang relevan secara tekstual tetapi tetap membutuhkan banyak bahan yang belum dimiliki.

Selain itu, pengguna dapat memiliki batasan seperti:
* `maximum 500 calories`
* `around 400 calories`
* `without cheese`
* `no milk`

Sistem perlu memahami *constraint* tersebut dan menghasilkan rekomendasi yang mempertimbangkan kondisi *pantry* aktual pengguna.

---

## 5. Product Objective

Sistem harus dapat:
1. Memahami *ingredient* dan *constraint* dari *natural-language query*;
2. Mencari *candidate recipes* yang relevan;
3. Menghitung seberapa banyak *ingredient* resep yang sudah tersedia;
4. Mengidentifikasi *ingredient* yang masih kurang;
5. Mempertimbangkan target kalori;
6. Menghindari *ingredient* yang secara eksplisit dilarang;
7. Melakukan *reranking candidate recipes*;
8. Menghasilkan rekomendasi *Top-K* yang *explainable*.

---

## 6. Non-Goals

Versi awal sistem tidak bertujuan untuk:
* Menghasilkan resep baru;
* Memodifikasi *cooking instructions*;
* Mengganti *ingredient* secara otomatis;
* Melakukan *medical diet recommendation*;
* Menjamin resep aman untuk *food allergy*;
* Menghitung nutrisi lengkap seperti protein, sodium, vitamin, dan mineral;
* Menggunakan GPT, Gemini, Claude, atau *frontier LLM API*;
* Melakukan *conversational multi-turn recommendation*.

*Catatan: Excluded ingredient hanya diperlakukan sebagai preference constraint, bukan medical allergy detection.*

---

## 7. Target User

Target utama pengguna sistem ini adalah:
* Mahasiswa;
* Pengguna yang memasak sendiri;
* Pengguna yang ingin menghabiskan bahan yang tersedia di dapur (*pantry*);
* Pengguna yang ingin mengurangi pembelian bahan tambahan;
* Pengguna yang memiliki *approximate calorie target*.

---

## 8. Primary User Stories

### User Story 1 — Pantry-Based Recommendation
> **Sebagai** pengguna, **saya ingin** memasukkan bahan yang saya miliki, **sehingga** sistem dapat menunjukkan resep yang paling memungkinkan dibuat tanpa harus membeli banyak bahan tambahan.

* **Example Input:** `I have: chicken, potato, carrot, onion, garlic`
* **Expected Output:**
  * **Recipe A:** Ingredient coverage: 90%, Missing: `celery`
  * **Recipe B:** Ingredient coverage: 78%, Missing: `butter, cream`

### User Story 2 — Calorie Target
> **Sebagai** pengguna, **saya ingin** menentukan *approximate calorie target*, **sehingga** resep yang direkomendasikan mendekati kebutuhan kalori saya.

* **Example Input:** `I have chicken, potato and carrot. I want something around 500 calories.`

### User Story 3 — Ingredient Exclusion
> **Sebagai** pengguna, **saya ingin** menyebutkan bahan yang tidak ingin digunakan.

* **Example Input:** `I have chicken, tomato, onion and garlic. Under 600 calories. No cheese or milk.`
* **Expected Behavior:** Sistem harus memberikan penalti atau mengecualikan resep yang mengandung `cheese` dan `milk`.

---

## 9. Dataset

### 9.1 Primary Dataset: RecipeNLG
**RecipeNLG** digunakan sebagai korpus resep utama.
* **Relevant Information:** `title`, `ingredients`, `directions`, `NER`
* **Fields Utama untuk Rekomendasi:** `title`, `ingredients`, `NER / cleaned ingredients`, `directions`

*Catatan: `directions` tidak digunakan untuk perhitungan similarity, tetapi akan ditampilkan setelah resep dipilih.*

---

## 10. Supporting Data

Sistem menggunakan data pendukung yang sama dengan notebook referensi:

### 10.1 Calorie Reference Dataset
Digunakan untuk mendapatkan pemetaan kalori (`ingredient` $\rightarrow$ `calories_per_100g`):
* `chicken breast` $\rightarrow$ `xxx kcal / 100 g`
* `potato` $\rightarrow$ `xxx kcal / 100 g`
* `butter` $\rightarrow$ `xxx kcal / 100 g`

### 10.2 Supplement Ingredient Table
Jika *ingredient* tidak tersedia di *calorie reference* utama, sistem menggunakan *supplement table*.

### 10.3 Zero-Calorie / Negligible-Calorie Ingredient List
Beberapa *ingredient* diperlakukan sebagai nol atau kalori yang dapat diabaikan sesuai *preprocessing* pada notebook referensi.

---

## 11. High-Level Architecture

```text
USER QUERY
    │
    ▼
┌───────────────────────────────────────┐
│ PIPELINE 1                            │
│ Query & Constraint Understanding      │
└───────────────────────────────────────┘
    │
    ▼  (Structured Query)
┌───────────────────────────────────────┐
│ PIPELINE 2                            │
│ Candidate Recipe Retrieval            │
└───────────────────────────────────────┘
    │
    ▼  (Candidate Recipes)
┌───────────────────────────────────────┐
│ PIPELINE 3                            │
│ Multi-Criteria Recipe Reranking       │
└───────────────────────────────────────┘
    │
    ▼
TOP-K RECIPES
```

---

## 12. Pipeline 1 — Query & Constraint Understanding

### 12.1 Objective
Mengubah *natural-language user query* menjadi *structured query*.

* **Example Input:** `"I have chicken, potato, carrot, onion and garlic. I want something around 500 calories without cheese."`
* **Target Representation:**
```json
{
  "available_ingredients": [
    "chicken",
    "potato",
    "carrot",
    "onion",
    "garlic"
  ],
  "excluded_ingredients": [
    "cheese"
  ],
  "calorie_target": 500,
  "calorie_min": null,
  "calorie_max": null
}
```

### 12.2 Subprocesses
```text
Raw Query 
  ↓
Text Normalization 
  ↓
Ingredient Extraction 
  ↓
Ingredient Normalization 
  ↓
Negative Constraint Extraction 
  ↓
Calorie Constraint Extraction 
  ↓
Structured Query
```

### 12.3 Ingredient Extraction
Baseline mengikuti pendekatan notebook referensi:
```text
User query 
  ↓
Lowercase 
  ↓
Regex tokenization 
  ↓
Lemmatization 
  ↓
Known ingredient vocabulary matching
```
*Ingredient vocabulary* dibangun dari korpus `RecipeNLG NER / cleaned ingredient`.
* **Example:** `"I've got chickens, potatoes and some carrots"` $\rightarrow$ `["chicken", "potato", "carrot"]`

### 12.4 Ingredient Normalization
Normalisasi diperlukan agar variasi bentuk kata tidak dianggap sebagai bahan yang berbeda:
* `potatoes` $\rightarrow$ `potato`
* `tomatoes` $\rightarrow$ `tomato`
* `eggs` $\rightarrow$ `egg`
* **Tools:** `NLTK WordNetLemmatizer`, `regex`, *custom ingredient dictionary*.

### 12.5 Negative Constraint Extraction
Pola yang didukung pada MVP: `no X`, `without X`, `exclude X`, `don't want X`.
* **Examples:**
  * `"without cheese"` $\rightarrow$ `excluded = ["cheese"]`
  * `"no milk or butter"` $\rightarrow$ `excluded = ["milk", "butter"]`
* **Implementation:** *Regex* + *ingredient vocabulary matching* (tidak membutuhkan LLM).

### 12.6 Calorie Constraint Extraction
Sistem membedakan setidaknya tiga bentuk *constraint*:
1. **Maximum:** `"under 500 calories"`, `"below 500 calories"`, `"max 500 calories"`
   ```json
   { "calorie_max": 500 }
   ```
2. **Range:** `"between 400 and 600 calories"`
   ```json
   { "calorie_min": 400, "calorie_max": 600 }
   ```
3. **Target:** `"around 500 calories"`, `"about 500 calories"`
   ```json
   { "calorie_target": 500 }
   ```

### 12.7 Tech Stack
`Python`, `Pandas`, `Regex`, `NLTK`, `WordNetLemmatizer`, *Custom ingredient vocabulary* (tanpa model generatif).

---

## 13. Offline Recipe Enrichment

Sebelum Pipeline 2 dijalankan, dataset RecipeNLG diproses secara *offline* (bukan *user-facing pipeline*):
```text
RecipeNLG 
  ↓
Ingredient Cleaning 
  ↓
Ingredient Parsing 
  ↓
Canonical Ingredient Extraction 
  ↓
Calorie Mapping 
  ↓
Total Calorie Estimation 
  ↓
Processed Recipe Corpus
```

---

## 14. Ingredient Parsing

*Raw ingredients* yang berbentuk seperti `2 tbsp butter`, `1/2 cup milk`, atau `200 g chicken breast` diparsing menjadi representasi terstruktur:
```json
{
  "quantity": 2,
  "unit": "tbsp",
  "ingredient": "butter"
}
```
Pendekatan *regex-based parsing* dari notebook referensi untuk kuantitas, satuan, dan nama bahan tetap dipertahankan.

---

## 15. Ingredient Matching to Calorie Database

Prioritas pencocokan bahan hasil *parsing* ke database referensi kalori:
```text
Exact Match 
  ↓
Normalized Match 
  ↓
Fuzzy Match 
  ↓
Supplement Table
```
* **Example:** `"unsalted butter"` $\rightarrow$ `butter` $\rightarrow$ *calorie reference*.

---

## 16. Recipe Calorie Estimation

Jika referensi nutrisi diberikan dalam $\text{kcal}/100\text{ gram}$, maka kalori per bahan ($C_i$) dan total kalori resep ($C_{\text{recipe}}$) dihitung dengan rumus:

$$C_i = \frac{W_i}{100} \times C_{100g, i}$$

$$C_{\text{recipe}} = \sum_{i=1}^{n} C_i$$

**Output Enrichment:**
```json
{
  "recipe_id": 123,
  "title": "Chicken Potato Soup",
  "ingredients": ["..."],
  "canonical_ingredients": ["..."],
  "total_calories": 482
}
```
*Catatan: Nilai kalori bersifat estimasi (estimated), bukan informasi nutrisi yang otoritatif.*

---

## 17. Pipeline 2 — Candidate Recipe Retrieval

### 17.1 Objective
Mengambil *candidate recipes* yang secara tekstual / *ingredient-wise* relevan terhadap bahan milik pengguna. Pipeline ini berfokus pada **high recall** dan belum menentukan rekomendasi akhir.

### 17.2 Text Representation
* **Corpus:** *Cleaned recipe ingredients*
* **Baseline:** TF-IDF
* **Konfigurasi Awal:** `TfidfVectorizer(max_features=1000, ngram_range=(1,2))` (dapat di-*tuning* selama eksperimen).

### 17.3 Query Vectorization
Bahan pengguna (misalnya: `"chicken potato carrot onion garlic"`) diubah menjadi vektor kueri TF-IDF ($q$) untuk dicocokkan dengan vektor TF-IDF resep ($r$).

### 17.4 Similarity Scoring
Relevansi kandidat dihitung menggunakan *cosine similarity*:

$$\text{CosSim}(q, r) = \frac{q \cdot r}{\|q\| \|r\|}$$

Top-$N$ kandidat (misalnya Top 50 dari total 1.000.000 resep) kemudian diteruskan ke Pipeline 3.

### 17.5 Why Candidate Retrieval Is Separate
Melakukan *multi-criteria scoring* terhadap seluruh korpus RecipeNLG kurang efisien. Karena itu digunakan pendekatan *two-stage retrieval*:
```text
Large Recipe Corpus 
  ↓  (Fast Retrieval - Pipeline 2: optimasi retrieval speed & recall)
Small Candidate Set 
  ↓  (Detailed Reranking - Pipeline 3: optimasi recommendation quality)
Top-K Recipes
```

### 17.6 Tech Stack
`Python`, `Pandas`, `Scikit-learn` (`TfidfVectorizer`, `Cosine Similarity`, opsional `NearestNeighbors`), `SciPy sparse matrix`, `Pickle / Joblib`.

---

## 18. Pipeline 3 — Pantry-Aware Multi-Criteria Reranking

### 18.1 Objective
Melakukan *reranking* terhadap *candidate recipes* berdasarkan tingkat kegunaan (*usability*) resep bagi pengguna. Pipeline inilah yang menjadi diferensiasi utama terhadap notebook referensi.

---

## 19. Ingredient Coverage Score

Misalkan $U$ adalah himpunan *ingredient* yang dimiliki pengguna, dan $R$ adalah himpunan *ingredient* yang diperlukan oleh resep. Skor cakupan bahan didefinisikan sebagai:

$$S_{\text{coverage}} = \frac{|U \cap R|}{|R|}$$

* **Example:**
  * **User ($U$):** `chicken`, `potato`, `carrot`, `onion`, `garlic`
  * **Recipe ($R$):** `chicken`, `potato`, `carrot`, `onion`, `garlic`, `celery`
  * **Coverage:** $5 / 6 = 0.833$ ($83.3\%$)

---

## 20. Missing Ingredient Score

Himpunan bahan yang kurang ($M$) adalah selisih himpunan bahan resep terhadap bahan pengguna:

$$M = R - U$$

* **Example:** Jika resep membutuhkan `celery` dan `cream` yang tidak dimiliki pengguna, maka $|M| = 2$.
* **Normalized Penalty:**
  $$P_{\text{missing}} = \frac{|R - U|}{|R|}$$
  Semakin banyak bahan yang hilang, semakin besar nilai penaltinya.

---

## 21. Pantry Staple Handling

Tidak semua bahan memiliki tingkat kepentingan yang sama. Bahan dapur umum (*common pantry ingredients*) seperti `salt`, `water`, `pepper`, dan `oil` tidak seharusnya memberikan penalti sebesar bahan utama jika tidak disebutkan oleh pengguna.

**Contoh Bobot Bahan ($w_i$):**

| Ingredient | Weight ($w_i$) |
| :--- | :---: |
| `chicken` | 1.00 |
| `potato` | 1.00 |
| `cream` | 0.80 |
| `salt` | 0.15 |
| `pepper` | 0.15 |
| `water` | 0.05 |

**Weighted Coverage:**

$$S_{\text{coverage}} = \frac{\sum_{i \in U \cap R} w_i}{\sum_{i \in R} w_i}$$

*Kamus pantry staple dibangun secara manual dari daftar bahan yang sangat umum.*

---

## 22. Calorie Compatibility Score

### 22.1 Maximum Calorie
Jika pengguna memilih batasan maksimum (misal: `under 500 calories`), resep dengan $C_{\text{recipe}} > 500$ dapat difilter langsung atau menerima penalti besar. Untuk MVP, **hard filtering** direkomendasikan.

### 22.2 Target Calorie
Untuk batasan target aproksimatif (misal: `around 500 calories`), digunakan kedekatan kalori (*calorie proximity*):

$$D_c = |C_{\text{recipe}} - C_{\text{target}}|$$

$$S_{\text{calorie}} = \max\left(0, 1 - \frac{|C_r - C_t|}{C_t}\right)$$

* **Example (Target $C_t = 500\text{ kcal}$):**
  * Recipe A ($490\text{ kcal}$) $\rightarrow$ skor tertinggi ($0.98$)
  * Recipe B ($450\text{ kcal}$) $\rightarrow$ skor ($0.90$)
  * Recipe C ($300\text{ kcal}$) $\rightarrow$ skor terendah ($0.60$)

---

## 23. Excluded Ingredient Penalty

Misalkan $E$ merupakan himpunan *excluded ingredients*, maka jumlah pelanggaran adalah:

$$\text{Violation} = |E \cap R|$$

Untuk MVP diterapkan *hard exclusion*:
$$\text{Violation} > 0 \implies \text{Recipe removed}$$

* **Example:** User meminta `"without cheese"`. Resep yang mengandung `chicken, tomato, cheese` akan langsung dieksklusi (*EXCLUDED*). Mekanisme *fallback* apabila kandidat terlalu sedikit dapat dipertimbangkan pada versi selanjutnya.

---

## 24. Final Recommendation Score

Fungsi skoring akhir yang diusulkan (dengan *hard exclusion* untuk bahan terlarang):

$$\text{Score}(r, q) = \alpha S_{\text{tfidf}} + \beta S_{\text{coverage}} + \gamma S_{\text{calorie}} - \delta P_{\text{missing}}$$

**Rekomendasi Bobot Awal:**

| Komponen | Parameter | Bobot Awal |
| :--- | :---: | :---: |
| TF-IDF similarity | $\alpha$ | `0.30` |
| Ingredient coverage | $\beta$ | `0.35` |
| Calorie compatibility | $\gamma$ | `0.20` |
| Missing penalty | $\delta$ | `0.15` |

*Catatan: Bobot ini tidak dianggap final dan nilainya harus divalidasi melalui eksperimen.*

---

## 25. Ranking Example

* **User Query:** `"I have chicken, potato, carrot, onion and garlic. Around 500 calories. No cheese."`
* **Evaluasi Candidate A:**
  * $S_{\text{tfidf}} = 0.78$
  * $S_{\text{coverage}} = 0.90$
  * $S_{\text{calorie}} = 0.96$
  * $P_{\text{missing}} = 0.10$
  * Excluded violation = `false`
* **Perhitungan Akhir:**
  $$\text{Score} = 0.30(0.78) + 0.35(0.90) + 0.20(0.96) - 0.15(0.10) \approx 0.726$$

Seluruh *candidate recipes* kemudian diurutkan berdasarkan *final score* tersebut.

---

## 26. Final Output Format

Tampilan output minimal untuk pengguna (Default Top-$K = 3$, Opsional Top-$K = 5$):

```text
#1 Chicken Potato Soup
Overall Match       : 91%
Ingredient Coverage : 86%
Estimated Calories  : 482 kcal

YOU HAVE:
✓ chicken
✓ potato
✓ carrot
✓ onion
✓ garlic

MISSING:
✗ celery

CONSTRAINTS:
✓ approximately 500 kcal
✓ no excluded ingredients

INGREDIENTS: [...]
DIRECTIONS: [...]
```

---

## 27. Recommendation Explainability

Sistem harus menjelaskan mengapa sebuah resep direkomendasikan dengan menampilkan faktor-faktor berikut:
* *Overall score*
* *Ingredient coverage*
* *Estimated calories*
* *Available ingredients*
* *Missing ingredients*
* *Excluded ingredient status*

Sistem tidak boleh hanya menampilkan skor mentah seperti `Similarity = 0.82` karena nilai tersebut sulit dipahami oleh *end user*.

---

## 28. Functional Requirements

* **FR-01 — Natural Language Query:** Sistem harus menerima masukan berupa *natural-language query*.
* **FR-02 — Ingredient Extraction:** Sistem harus mengidentifikasi *available ingredients* dari kueri.
* **FR-03 — Ingredient Normalization:** Sistem harus menormalisasi bentuk kata *ingredient* sebelum perbandingan.
* **FR-04 — Calorie Constraint:** Sistem harus mengenali *maximum calorie*, *calorie range*, dan *approximate calorie target*.
* **FR-05 — Negative Ingredient Constraint:** Sistem harus mendukung pengecualian bahan (*ingredient exclusions*).
* **FR-06 — Recipe Retrieval:** Sistem harus mengambil resep relevan dari korpus RecipeNLG.
* **FR-07 — Ingredient Coverage:** Sistem harus menghitung *pantry coverage* untuk kandidat resep.
* **FR-08 — Missing Ingredient Detection:** Sistem harus mengembalikan daftar bahan yang dibutuhkan resep tetapi tidak dimiliki pengguna.
* **FR-09 — Calorie Estimation:** Sistem harus menyediakan estimasi kalori resep.
* **FR-10 — Multi-Criteria Ranking:** Sistem harus mengurutkan resep menggunakan kriteria yang lebih luas daripada sekadar kemiripan leksikal.
* **FR-11 — Explainable Recommendation:** Setiap rekomendasi harus menampilkan faktor-faktor pendukung rekomendasi secara transparan.

---

## 29. Non-Functional Requirements

* **NFR-01 — No Frontier LLM Dependency:** Sistem inti harus beroperasi tanpa bergantung pada OpenAI API, Gemini API, Claude API, atau layanan *frontier LLM* serupa.
* **NFR-02 — Reproducibility:** *Preprocessing*, vektorisasi, dan konfigurasi skoring harus bersifat *reproducible*.
* **NFR-03 — Offline Processing:** Proses komputasi berat seperti pengayaan kalori (*calorie enrichment*), *TF-IDF fitting*, dan normalisasi resep harus dijalankan secara *offline*. Proses rekomendasi saat *runtime* hanya menggunakan artefak yang telah diproses sebelumnya.
* **NFR-04 — Explainability:** Komponen *ranking* akhir harus transparan dan dapat diinspeksi (*inspectable*).

---

## 30. Evaluation Strategy

Evaluasi tidak hanya mengandalkan skor *cosine similarity*. Ketiga pipeline dalam sistem harus dievaluasi secara terpisah maupun terintegrasi.

---

## 31. Pipeline 1 Evaluation

### 31.1 Ingredient Extraction
Dievaluasi menggunakan *annotated test queries* dengan metrik **Precision**, **Recall**, dan **F1-score**.
* **Query:** `"I have chicken potatoes and garlic"`
* **Ground Truth:** `["chicken", "potato", "garlic"]`

### 31.2 Constraint Extraction
Dievaluasi menggunakan:
* **Calorie Extraction Accuracy**
* **Excluded Ingredient Extraction F1**

---

## 32. Pipeline 2 Evaluation

*Recommendation retrieval* dievaluasi menggunakan *synthetic query generation*:
1. Ambil sampel resep dari *test set*;
2. Ambil sebagian daftar bahan dari resep tersebut;
3. Gunakan potongan bahan tersebut sebagai *simulated pantry*;
4. Jalankan proses *retrieval*;
5. Periksa posisi resep asli atau resep terkait pada daftar kandidat.

**Metrik Evaluasi:** `Recall@K`, `Precision@K`, `MRR` (*Mean Reciprocal Rank*).

---

## 33. Pipeline 3 Evaluation

**Metrik Utama Evaluasi:**
* `Ingredient Coverage@K` (Target: $\uparrow$)
* `Average Missing Ingredients@K` (Target: $\downarrow$)
* `Calorie Distance` (Target: $\downarrow$)
* `Constraint Violation Rate` (Target: $\downarrow$)

---

## 34. Baseline vs Proposed System

Evaluasi wajib membandingkan sistem yang diusulkan (*Proposed*) dengan *baseline* bergaya notebook referensi (*TF-IDF + Cosine Similarity* saja):

| Metric | Baseline (TF-IDF Only) | Proposed System |
| :--- | :---: | :---: |
| `Ingredient Coverage@3` | TBD | TBD |
| `Missing Ingredients@3` | TBD | TBD |
| `Calorie Distance` | TBD | TBD |
| `Constraint Violation` | TBD | TBD |
| `MRR` | TBD | TBD |

---

## 35. Ablation Study

Untuk menunjukkan kontribusi masing-masing komponen, dilakukan eksperimen *ablation* bertahap:
* **Model A:** `TF-IDF only`
* **Model B:** `TF-IDF` + `Ingredient Coverage`
* **Model C:** `TF-IDF` + `Ingredient Coverage` + `Missing Ingredient Penalty`
* **Model D (Full):** `TF-IDF` + `Ingredient Coverage` + `Missing Ingredient Penalty` + `Calorie Score` + `Exclusion Filter`

Dengan demikian dapat diketahui fitur mana yang memberikan peningkatan performa terbesar.

---

## 36. Success Criteria

MVP dianggap berhasil apabila:
1. Ekstraksi bahan dari bahasa alami dapat bekerja dengan baik pada *manually annotated test queries*;
2. *Calorie constraint* dapat diidentifikasi secara andal (*reliable*);
3. Sistem berhasil melakukan *retrieval* resep dari dataset RecipeNLG;
4. Rekomendasi menampilkan informasi *ingredient coverage*;
5. Rekomendasi menampilkan informasi *missing ingredients*;
6. Estimasi kalori tersedia untuk sebagian besar *candidate recipes*;
7. Bahan terlarang (*forbidden ingredient*) tidak muncul pada rekomendasi akhir;
8. *Proposed reranker* mengungguli *baseline* TF-IDF setidaknya pada metrik yang berorientasi pada *pantry*.

---

## 37. MVP Scope

Untuk mengendalikan kompleksitas, ruang lingkup MVP dibatasi pada:
* **Query:** *Available ingredients*, *excluded ingredients*, dan *calorie target / maximum*.
* **Retrieval:** TF-IDF.
* **Reranking:** *Ingredient coverage*, *cosine similarity*, *missing ingredient penalty*, *calorie compatibility*, dan *excluded ingredient filter*.
* **Output:** *Top-3 recipes* lengkap dengan *match score*, estimasi kalori, *coverage*, bahan tersedia, bahan kurang, dan instruksi memasak (*directions*).

---

## 38. Out of Scope for MVP

Fitur yang **tidak** termasuk dalam MVP:
* Klasifikasi nutrisi diet medis;
* Substitusi bahan otomatis (*ingredient substitution*);
* Generasi resep baru (*recipe generation*);
* Optimasi makronutrisi (protein/lemak/karbohidrat);
* Pemahaman gambar makanan (*food image understanding*);
* Riwayat pengguna yang dipersonalisasi (*personalized user history*);
* *Collaborative filtering*;
* Klasifikasi emosi (*emotion classification*);
* *LLM chatbot*.

---

## 39. Tech Stack

* **Core Language:** Python
* **Data Processing:** Pandas, NumPy
* **NLP:** NLTK, Regex, WordNetLemmatizer, Custom dictionaries
* **Information Retrieval:** Scikit-learn (`TfidfVectorizer`, `Cosine Similarity`, `NearestNeighbors` opsional)
* **Serialization:** Pickle atau Joblib
* **Experiment Tracking:** Minimum menggunakan CSV/JSON; Opsional menggunakan MLflow atau Weights & Biases
* **Interface:** Direkomendasikan Streamlit; Alternatif Gradio

---

## 40. Proposed Repository Structure

```text
cook-from-pantry/
├── data/
│   ├── raw/
│   │   ├── recipenlg/
│   │   └── calories/
│   ├── processed/
│   │   ├── recipes.parquet
│   │   ├── ingredients.parquet
│   │   └── pantry_staples.csv
│   └── test/
│       └── annotated_queries.json
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_recipe_preprocessing.ipynb
│   ├── 03_calorie_enrichment.ipynb
│   ├── 04_query_understanding.ipynb
│   ├── 05_retrieval.ipynb
│   └── 06_reranking_evaluation.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── ingredient_parser.py
│   │   ├── ingredient_normalizer.py
│   │   └── calorie_mapper.py
│   ├── pipeline1/
│   │   ├── ingredient_extractor.py
│   │   ├── constraint_extractor.py
│   │   └── query_parser.py
│   ├── pipeline2/
│   │   ├── vectorizer.py
│   │   └── retriever.py
│   └── pipeline3/
│       ├── coverage.py
│       ├── calorie_score.py
│       ├── reranker.py
│       └── recommender.py
├── artifacts/
│   ├── tfidf_vectorizer.pkl
│   ├── recipe_vectors.pkl
│   └── ingredient_vocabulary.pkl
├── evaluation/
│   ├── evaluate_extraction.py
│   ├── evaluate_retrieval.py
│   └── evaluate_ranking.py
├── app/
│   └── streamlit_app.py
├── tests/
├── requirements.txt
├── README.md
└── PRD.md
```

---

## 41. Development Plan

* **Phase 1 — Dataset Understanding:** Memuat RecipeNLG, memeriksa format bahan, *NER field*, dataset kalori, *missing values*, dan mengukur kosakata bahan. *Output: Dataset profiling report.*
* **Phase 2 — Recipe Preprocessing:** Normalisasi bahan, *parsing* kuantitas & satuan, ekstraksi bahan inti, membangun *ingredient vocabulary*, dan mendefinisikan *pantry staples*. *Output: Processed recipe corpus.*
* **Phase 3 — Calorie Enrichment:** Pencocokan kalori bahan, *fuzzy matching*, konversi satuan, estimasi kalori resep, dan analisis cakupan (*% recipes with reliable estimated calorie*).
* **Phase 4 — Pipeline 1:** Ekstraksi bahan tersedia, bahan dikecualikan, kalori, pembentukan *structured query*, dan pembuatan *manual test set*.
* **Phase 5 — Pipeline 2:** *TF-IDF fitting*, vektorisasi kueri, *cosine similarity*, dan pengambilan *Top-N candidate retrieval* (direkomendasikan $N = 50$).
* **Phase 6 — Pipeline 3:** Pencocokan himpunan bahan, skoring cakupan, perhitungan bahan kurang, pembobotan *pantry staple*, kompatibilitas kalori, penanganan eksklusi, dan skoring berbobot akhir.
* **Phase 7 — Evaluation:** Menjalankan eksperimen *baseline* vs *proposed reranker*, studi ablasi (*ablation experiments*), pembuatan tabel metrik, contoh peringkat, dan analisis kesalahan (*error analysis*).
* **Phase 8 — Application:** Membangun antarmuka sederhana menggunakan Streamlit (`User Query` $\rightarrow$ `Recommend` $\rightarrow$ `Top-3 Cards`).

---

## 42. Risks and Mitigations

1. **Risk 1 — Ingredient Parsing Errors:** Frasa seperti `"1 can cream of mushroom soup"` mungkin tidak ter-parse dengan tepat.
   * *Mitigation:* Gunakan kombinasi *NER field*, regex, normalisasi, *fuzzy matching*, dan kamus pengecualian manual (*manual exception dictionary*).
2. **Risk 2 — Calorie Estimates Are Inaccurate:** Kuantitas dan satuan pada resep sering kali tidak lengkap.
   * *Mitigation:* Tampilkan nilai secara eksplisit sebagai **Estimated Calories** dan jangan mengklaimnya sebagai informasi nutrisi mutlak.
3. **Risk 3 — Ingredient Synonyms:** Bahan seperti `bell pepper` vs `capsicum` atau `scallion` vs `green onion` dapat dianggap berbeda.
   * *Mitigation:* Buat *alias dictionary* untuk memetakan sinonim ke *canonical ingredient*.
4. **Risk 4 — Pantry Staples Dominate Similarity:** Bahan umum seperti `salt`, `water`, `oil`, dan `pepper` muncul sangat sering.
   * *Mitigation:* Gunakan TF-IDF yang dikombinasikan dengan *pantry staple down-weighting* saat proses *reranking*.
5. **Risk 5 — TF-IDF Cannot Understand Semantic Equivalence:** Sinonim yang tidak terdaftar dapat memiliki jarak vektor yang jauh.
   * *Mitigation:* Normalisasi bahan dan kamus sinonim untuk MVP; *semantic embedding* dipertimbangkan untuk pengembangan selanjutnya.
6. **Risk 6 — Too Few Candidates After Constraints:** Aturan kalori dan eksklusi yang terlalu ketat dapat mengeliminasi seluruh kandidat resep.
   * *Mitigation:* Kembalikan pesan *"No exact recommendation found"*, lalu tampilkan opsi alternatif terdekat (*Closest alternatives*) dengan peringatan eksplisit bahwa sebagian batasan telah dilonggarkan.

---

## 43. Future Improvements

* **Semantic Retrieval:** Mengganti atau menggabungkan TF-IDF dengan model *Sentence Transformer* berukuran kecil (tetap tanpa menggunakan *generative LLM*).
* **Nutrition Expansion:** Menambahkan perhitungan makronutrisi seperti protein, karbohidrat, lemak, dan sodium.
* **Ingredient Substitution:** Memberikan saran pengganti bahan (misal: kekurangan `cream` $\rightarrow$ saran substitusi `milk`).
* **Recipe Diversity:** Menggunakan *Maximal Marginal Relevance* (MMR) untuk mengurangi hasil *Top-K* yang terlalu repetitif.
* **Learned-to-Rank:** Mengganti pembobotan manual saat ini dengan model *supervised ranking* apabila data berlabel yang sesuai telah tersedia.

---

## 44. Core Research Question

**Pertanyaan Penelitian Utama:**
> *Can pantry-aware multi-criteria reranking produce more practically useful recipe recommendations than TF-IDF cosine similarity alone?*

**Pertanyaan Pendukung:**
1. Apakah *ingredient coverage* meningkatkan kegunaan (*usability*) rekomendasi?
2. Apakah penalti *missing ingredient* mengurangi jumlah bahan tambahan yang perlu dibeli pengguna?
3. Dapatkah *calorie compatibility* diintegrasikan tanpa menurunkan relevansi resep secara signifikan?
4. Seberapa besar peningkatan performa yang diperoleh dari penggabungan *lexical retrieval* dengan pencocokan bahan terstruktur (*structured ingredient matching*)?

---

## 45. Final System Definition

**YummyTummy** didefinisikan sebagai:
> *Constraint-aware information retrieval and recommendation system that recommends recipes from RecipeNLG by jointly considering textual ingredient relevance, pantry availability, missing ingredients, calorie compatibility, and explicit ingredient exclusions.*

Sistem ini terdiri atas tiga *pipeline* utama NLP/IR:
1. **Pipeline 1:** *Query & Constraint Understanding*
2. **Pipeline 2:** *Candidate Recipe Retrieval*
3. **Pipeline 3:** *Pantry-Aware Multi-Criteria Reranking*

Kontribusi metodologis utama dari proyek ini bukanlah pembuatan resep (*recipe generation*), melainkan transformasi sistem temu kembali resep konvensional berbasis kemiripan teks menjadi sistem rekomendasi yang dapat dijelaskan (*explainable*), sadar batasan (*constraint-aware*), dan berorientasi pada ketersediaan bahan dapur (*pantry-oriented*) tanpa ketergantungan pada *generative LLMs*.