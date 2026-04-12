# Hệ thống AI gợi ý ẩm thực và tối ưu hóa ngân sách

**Bài tập lớn môn học:** Nhập môn Trí tuệ Nhân tạo (Introduction to AI)

**Mã môn học:** CO3061

**Học kỳ:** II — Năm học 2025–2026

**Trường:** Đại học Bách Khoa, ĐHQG-HCM

**Giảng viên hướng dẫn:** TS. Trương Vĩnh Lân

**GitHub repository:** [https://github.com/dvydinh/yumify](https://github.com/dvydinh/yumify)

---

## 1. Giới thiệu tổng quan

Dự án xây dựng một hệ thống Trí tuệ Nhân tạo tích hợp (Hybrid AI System), kết hợp AI cổ điển (Symbolic AI, Search, Constraint Satisfaction, Probabilistic Reasoning) và Học máy (Multinomial Naive Bayes) để giải quyết bài toán gợi ý thực đơn đa ràng buộc. Hệ thống tiếp nhận yêu cầu bằng ngôn ngữ tự nhiên tiếng Anh, phân tích ngữ nghĩa, và đề xuất công thức nấu ăn thỏa mãn đồng thời các ràng buộc về chi phí, lượng calo, và điều kiện y khoa của người dùng.

Khi nạp dataset Kaggle Food.com (5000 recipes qua Stratified Sampling), hệ thống bao phủ 17 nền ẩm thực nhờ cơ chế phát hiện cuisine 3 tầng (tag → recipe name → ingredient signature). Khi offline, fallback về tập local `recipes.json` (25 recipes, 6 cuisines).

---

## 2. Thông tin nhóm

| Họ và tên | MSSV | Email |
|-----------|------|-------|
| Đinh Đoàn Vy | 2353350 | vy.dinhdoan2005@hcmut.edu.vn |
| Trần Thiên Lộc | 2352715 | loc.tranthien3905@hcmut.edu.vn |
| Nguyễn Trần Gia Bảo | 2352104 | bao.nguyentrangia@hcmut.edu.vn |
| Dương Lê Nhật Duy | 2352171 | duy.duong250405@hcmut.edu.vn |

---

## 3. Nguồn dữ liệu

### 3.1. Dataset chính — Kaggle Food.com

Hệ thống sử dụng tập dữ liệu **"Food.com Recipes and User Interactions"** từ Kaggle (tác giả `shuyangli94`), quy mô hơn 230,000 công thức. Hệ thống tự động tải, parse, và trích xuất đặc trưng TF-IDF từ tập dữ liệu này khi chạy trên Colab thông qua hàm `download_recipe_dataset()`.

Các kỹ thuật xử lý dữ liệu Kaggle:

- **Heuristic Quantity Estimation:** Dataset Kaggle chỉ cung cấp tên nguyên liệu mà **không có định lượng** (khối lượng gram). Hàm `_estimate_ingredient_quantity()` áp dụng chuẩn USDA Standard Portion Sizes để gán khối lượng mặc định: Thịt 200g, Hải sản 150g, Rau củ 100g, Gia vị 5g, Tinh bột 150g.

- **Cross-reference Cost Computation:** Dataset Kaggle không có cột giá tiền. Hệ thống tính chi phí từng công thức bằng cách đối chiếu từng nguyên liệu với `ingredients.json` (exact match → substring match → category-based fallback), nhân giá theo khối lượng: `cost = price_per_kg × (qty_g / 1000)`.

- **Stratified Cuisine Sampling:** Hàm `_detect_cuisine_from_tags()` phát hiện cuisine 3 tầng (tag → recipe name patterns 120+ → ingredient signatures 80+). Kết hợp với quota-based sampling (`MAX_PER_CUISINE = max_recipes / 8 = 625`) để ngăn chặn class imbalance (nếu không, 99%+ dữ liệu rơi vào nhãn "International").

### 3.2. Dataset fallback — Local JSON

- `data/recipes.json`: 25 công thức, 6 cuisines (Italian, Japanese, Korean, Mexican, Vietnamese, Western).
- `data/ingredients.json`: 75 nguyên liệu với giá USD, VND, và calories.

### 3.3. API bổ trợ (tùy chọn)

- **Spoonacular API:** fallback data thời gian thực khi cần bổ sung dữ liệu. Yêu cầu API key riêng.

> **Lưu ý:** Pipeline AI chính (NLP → ML → KB → TF-IDF → A* → Bayes) hoạt động hoàn toàn offline trên dataset Kaggle/local. API chỉ đóng vai trò bổ sung tùy chọn, không nằm trong luồng xử lý cốt lõi.

---

## 4. Kiến trúc hệ thống và các trụ cột AI

Pipeline xử lý đầy đủ 6 bước:

```
User Input → NLP Parser → KB Forward Chaining → Filter unsafe recipes
                                                        ↓
                              ML Naive Bayes ← TF-IDF Similarity ranking
                                                        ↓
                                              Bayesian Risk (EU ranking) → Output
```

Hệ thống được xây dựng trên 5 trụ cột AI theo yêu cầu đề bài:

### 4.1. Học máy (Machine Learning) — L.O.3

**Pipeline NLP → ML (Data Flow):**

```
User Input → NLP Parser (NER) → Mảng thực thể ["beef", "noodle"]
                                        ↓
               Multinomial Naive Bayes (train trên 5000 Kaggle recipes)
                                        ↓
               P(Cuisine | features) = argmax → "Vietnamese" (conf=0.619)
```

User Input được xử lý qua module **NER** (Named Entity Recognition) trích xuất các thực thể nguyên liệu. Mảng thực thể này được đẩy vào mô hình **Multinomial Naive Bayes** (`ml_classifier.py`) để phân loại nền ẩm thực. Trên Colab, mô hình được train trên **5000 Kaggle recipes** (17 cuisines, 1792 vocabulary, qua stratified sampling). Khi chạy local, fallback về `recipes.json` (25 recipes, 6 cuisines).

Các thành phần chi tiết:

- **NLP Parser (NER):** Ontology-based Entity Extraction trích xuất thực thể từ văn bản tiếng Anh (nguyên liệu, ràng buộc y khoa, ngân sách). Bảng từ vựng 200+ mục.
- **TF-IDF Vectorization:** Biến đổi tập công thức thành không gian vector đặc trưng (7,780 terms trên Kaggle), cho phép Cosine Similarity ranking.
- **Naive Bayes Classifier:** Multinomial Naive Bayes với Laplace smoothing. Tính toán Posterior $P(C|W) \propto P(C) \cdot \prod P(w_i|C)$ với log-sum-exp normalization để tránh underflow. Accuracy: 5/8 (62.5%) trên 8 test cases tiêu biểu.

### 4.2. Ràng buộc (CSP) — L.O.1

Thuật toán Constraint Satisfaction Problem với **Backtracking**, **Forward Checking nâng cao**, và heuristic **MRV** (Minimum Remaining Values).

**Kỹ thuật cốt lõi — Dual-constraint Pruning với Bounds Propagation:**

Forward Checking thực hiện **lan truyền cận trên/dưới cho biến Calo** (Bounds Propagation). Trước khi duyệt candidate cho mỗi biến chưa gán, hệ thống tính:

- $min\_future\_cals$: tổng calo **tối thiểu** từ Domain của tất cả biến còn lại.
- $max\_future\_cals$: tổng calo **tối đa** từ Domain của tất cả biến còn lại.

Điều kiện cắt tỉa kép:
```
if (current_cal + candidate_cal + min_future_cals) > max_cal:
    prune()  # Chắc chắn vi phạm cận trên
if (current_cal + candidate_cal + max_future_cals) < min_cal:
    prune()  # Chắc chắn không thể đạt đủ cận dưới
```

### 4.3. Logic và tri thức (Knowledge Base) — L.O.2.1

Hệ chuyên gia dựa trên luật sử dụng Propositional Logic. Tri thức được biểu diễn dưới dạng `Fact` (mệnh đề nguyên tử) và `HornClause`. Forward Chaining Engine suy diễn tiến để loại bỏ nguyên liệu vi phạm điều kiện y khoa.

7 rules trong `rules.json`:

| Rule | Condition | Nguyên liệu bị loại |
|------|-----------|---------------------|
| R_STOMACH | stomachache | chili, garlic, lemon, tomato |
| R_GOUT | gout | beef, mushroom, seafood, shrimp, squid |
| R_DIABETES | diabetes | sugar |
| R_HBP | hypertension | cheese, fish sauce, salt |
| R_PREGNANCY | pregnancy | green papaya, sashimi |
| R_SEAFOOD_ALLERGY | seafood allergy | crab, fish, shrimp, squid |
| R_VEGETARIAN | vegetarian | beef, chicken, fish, pork, shrimp |

**Key Mapping Layer:** NLP output được normalize trước khi truyền vào KB (ví dụ: `hypertension` → `condition:high blood pressure`, `vegetarian` → `preferences:vegetarian`) để đảm bảo 7/7 rules fire chính xác.

### 4.4. Tìm kiếm (Search) — L.O.1

Thuật toán **A* (A-Star Search)** với **Admissible Heuristic** cho bài toán N-Days Meal Planning.

Không gian trạng thái:

- **State:** `(day, total_cost, total_calories, sorted_multiset_of_recipes)`
- **Action:** chọn một công thức cho ngày tiếp theo.
- **Goal:** đủ N ngày, tổng chi phí ≤ budget, calo trong khoảng cho phép.
- **Cost:** tổng chi phí tích lũy $g(n)$ (bao gồm repetition penalty); heuristic $h(n) = (N - day) \times min\_cost$ (Admissible vì không bao giờ đánh giá quá thực tế).

**Kỹ thuật cốt lõi:**

1. **Canonical State Representation:** Vì Goal Test chỉ kiểm tra tổng chi phí và tổng calo (không quan tâm thứ tự), các hoán vị của cùng một tập công thức là trạng thái tương đương. Hệ thống sắp xếp `selected_recipes` trước khi nạp vào tập visited, giảm kích thước cây tìm kiếm:

$$O(R^N) \rightarrow O\left(\frac{R^N}{N!}\right) = O(C(R,N))$$

2. **Repetition Penalty:** Khi một công thức đã xuất hiện trong plan, việc chọn lại nó bị phạt theo cấp số nhân: `penalty = cost × 2^repeat_count`. Nếu lặp ngày liên tiếp, cộng thêm `cost × 3.0`. Đảm bảo A* ưu tiên đa dạng thực đơn mà vẫn cho phép lặp khi ngân sách buộc phải vậy.

**Kết quả:** 5/5 test cases PASS. 3/3 unique (3 ngày), 5/5 unique (5 ngày), 6/7 unique (7 ngày). Max thời gian: 60 ms.

### 4.5. Xác suất (Bayesian Network) — L.O.2.2

Expert-driven Bayesian Network sử dụng Subjectivist Probabilities (xác suất gán bởi chuyên gia dinh dưỡng theo phương pháp Expert Elicitation — de Finetti). Mô hình gồm hai tầng inference:

1. **Posterior Preference Inference:** $P(Like | e_1, e_2, \ldots, e_n)$ tính qua Bayes' Rule trong log-space với log-sum-exp normalization.

2. **Extended Noisy-OR Model with Synergy Hidden Causes:**

   Mô hình cơ bản Noisy-OR giả định các yếu tố rủi ro là **độc lập**: $P(Risk) = 1 - \prod(1 - P_i)$. Trong thực tế dinh dưỡng, một số yếu tố có **tác động hiệp đồng** khi xuất hiện đồng thời (ví dụ: chili + oil gây kích ứng mạnh hơn tổng riêng lẻ).

   Thay vì nhân xác suất với hằng số (vi phạm tiên đề Kolmogorov), hệ thống mô hình hóa synergy như một **hidden cause** bổ sung trong Noisy-OR:

   $$P(Risk) = 1 - \prod_{i}(1 - P_i) \times \prod_{j}(1 - P_{synergy_j})$$

   CPT keys hoàn toàn bằng tiếng Anh (`chili`, `oil`, `milk`, `shrimp`, v.v.) để khớp với ingredient names trong Kaggle dataset. Bao gồm 24 risk factors cá nhân + 8 synergy pairs + health-conditional CPT cho stomachache, gout, diabetes.

   **Kết quả:** Spicy Shrimp Oil Pasta (stomachache): P(Risk) = 0.9970 (9 factors). Steamed Vegetables: P(Risk) = 0.0000.

   *Tham khảo: Henrion (1989), "Some Practical Issues in Constructing Belief Networks", UAI'89.*

3. **Expected Utility Theory:** $EU = P(Like) - \lambda \cdot P(Risk)$ ($\lambda = 0.3$). Kết hợp Posterior Preference và Risk Assessment thành Expected Utility để ra quyết định tối ưu trong môi trường bất định.

---

## 5. Cấu trúc thư mục

```
yumify/
├── notebooks/            # Front-end Google Colab
│   └── main_frontend.ipynb   # Notebook chính — Runtime → Run All
├── modules/              # Các module AI cốt lõi (7 modules)
│   ├── __init__.py
│   ├── nlp_parser.py         # NLP Parser (NER + ML Naive Bayes tích hợp)
│   ├── ml_classifier.py      # Multinomial Naive Bayes Classifier
│   ├── search_engine.py      # A* Search (Canonical State + Repetition Penalty)
│   ├── csp_solver.py         # CSP Solver (Dual-constraint Pruning)
│   ├── knowledge_base.py     # Forward Chaining Engine + Key Mapping Layer
│   └── bayes_risk.py         # Bayesian Network (Extended Noisy-OR + Synergy)
├── features/             # Trích xuất đặc trưng và kết nối dữ liệu
│   ├── __init__.py
│   ├── api_client.py         # Kết nối Spoonacular API (tùy chọn)
│   ├── data_loader.py        # Tải và parse dữ liệu JSON/CSV
│   ├── data_downloader.py    # Tải dữ liệu bổ sung
│   └── feature_extractor.py  # TF-IDF + Kaggle parser + Stratified Sampling
├── data/                 # Dữ liệu cục bộ
│   ├── recipes.json          # 25 công thức (6 cuisines, có cost + calories)
│   ├── ingredients.json      # 75 nguyên liệu (price_usd, price_vnd, calories)
│   ├── rules.json            # 7 luật suy diễn y khoa (Forward Chaining)
│   └── bayes_cpt.json        # CPT tables cho Bayesian Network
├── reports/              # Báo cáo
│   └── report.pdf            # Báo cáo PDF
└── README.md
```

---

## 6. Hướng dẫn cài đặt và vận hành

Dự án được thiết kế để chạy hoàn toàn trên **Google Colab**.

### Các bước thực hiện

1. Mở file `notebooks/main_frontend.ipynb` trên Google Colab.
2. Chạy **Runtime → Run all**.
3. Cell đầu tiên tự động clone mã nguồn từ GitHub và cài đặt thư viện.
4. Cell tiếp theo tự động tải dataset Food.com (5000 recipes), parse, và trích xuất TF-IDF features (7,780 terms).
5. Các cell còn lại chạy pipeline và kiểm thử từng module.

Toàn bộ quá trình hoàn toàn tự động, không cần thao tác thủ công. Input bằng **tiếng Anh**.

### Yêu cầu thư viện

```
numpy
requests
```

Tất cả được cài đặt tự động trong notebook.

---

## 7. Liên kết

- Báo cáo PDF: `reports/report.pdf`
- Colab notebook: [Mở trên Colab](https://colab.research.google.com/drive/15rI6G4KB-Es5MD-mUS2iCP1GFIQ2c1Iu?usp=sharing)
- GitHub repository: [https://github.com/dvydinh/yumify](https://github.com/dvydinh/yumify)

---

## 8. Tài liệu tham khảo

1. S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th Edition, Pearson, 2020. — Chương 3–4 (Search), Chương 6 (CSP), Chương 7 (Logic), Chương 13–16 (Probability & Bayes), Chương 20 (Naive Bayes).
2. P.E. Hart, N.J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 1968.
3. M. Henrion, "Some Practical Issues in Constructing Belief Networks", *Proceedings of the 3rd Conference on Uncertainty in AI (UAI)*, 1989. — Noisy-OR extensions and hidden causes.
4. G. Salton, C. Buckley, "Term-weighting approaches in automatic text retrieval", *Information Processing & Management*, 1988.
5. Kaggle Dataset: shuyangli94, "Food.com Recipes and User Interactions", https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions
