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

Hệ thống bao phủ 6 nền ẩm thực: Việt Nam, Ý, Nhật Bản, Hàn Quốc, Mexico, và Phương Tây.

---

## 2. Thông tin nhóm

| Họ và tên | MSSV | Email |
|-----------|------|-------|
| Đinh Đoàn Vy | 2353350 | vy.dinhdoan2005@hcmut.edu.vn |
| Trần Thiên Lộc | 2352715 | loc.tranthien3905@hcmut.edu.vn |
| Nguyễn Trần Gia Bảo | 2352103 | bao.nguyentrangia@hcmut.edu.vn |
| Dương Lê Nhật Duy | 2352171 | duy.duong250405@hcmut.edu.vn |

---

## 3. Nguồn dữ liệu

### 3.1. Dataset chính

Hệ thống sử dụng tập dữ liệu **"Food.com Recipes and User Interactions"** từ Kaggle (được công bố bởi tác giả `shuyangli94`). Đây là tập dữ liệu quy mô lớn với hơn 230,000 công thức, được sử dụng rộng rãi trong các nghiên cứu về Recommender Systems. Hệ thống tự động fetch, parse, và trích xuất đặc trưng TF-IDF từ tập dữ liệu này khi chạy trên Colab.

- **Heuristic Quantity Estimation (Giả lập khối lượng nguyên liệu):** Dataset Kaggle Food.com chỉ cung cấp danh sách tên nguyên liệu mà **không có định lượng** (khối lượng gram). Để đảm bảo các module AI tối ưu hóa trên dữ liệu có ý nghĩa toán học (thay vì `quantity = 1` tĩnh cho mọi nguyên liệu), nhóm đã xây dựng hàm heuristic `_estimate_ingredient_quantity()` áp dụng **chuẩn USDA Standard Portion Sizes** để gán khối lượng mặc định theo loại nguyên liệu: Thịt 200g, Hải sản 150g, Rau củ 100g, Gia vị 5g, Tinh bột 150g, v.v. Quyết định thiết kế này cho phép các module A* Search và CSP Solver tối ưu hóa chi phí và calo trên dữ liệu phản ánh thực tế dinh dưỡng, thay vì chạy trên dữ liệu hoàn toàn đồng nhất.

### 3.2. API bổ trợ (Extra Feature — ngoài pipeline chấm điểm chính)

- **Spoonacular API:** cung cấp fallback data thời gian thực, bao gồm thông tin dinh dưỡng và giá cả nguyên liệu khi dataset cục bộ không đáp ứng truy vấn.

> **Lưu ý:** Pipeline AI chính (NLP → ML → CSP → A* → Bayes) hoạt động hoàn toàn offline trên dataset cục bộ và các thuật toán cổ điển. API chỉ đóng vai trò bổ sung dữ liệu, không nằm trong luồng xử lý cốt lõi.

---

## 4. Kiến trúc hệ thống và các trụ cột AI

Pipeline xử lý tổng thể:

```
User Input → NLP Parser → TF-IDF Vectorization → CSP Solver → A* Search → Knowledge Base → Bayesian Risk → Output
```

Hệ thống được xây dựng trên 5 trụ cột AI theo yêu cầu đề bài:

### 4.1. Học máy (Machine Learning) — L.O.3

**Pipeline tích hợp NLP → ML (Data Flow thực tế):**

```
User Input → NLP Parser (NER) → Mảng thực thể ["beef", "noodle", "chili"]
                                         ↓
               Multinomial Naive Bayes (đã train trước trên recipes.json)
                                         ↓
               P(Cuisine | features) = argmax → "Vietnamese" (0.82 confidence)
```

User Input được xử lý qua module **NER** (Named Entity Recognition) để trích xuất các thực thể nguyên liệu. Sau đó, mảng thực thể này được đẩy **trực tiếp** vào mô hình **Multinomial Naive Bayes** (`ml_classifier.py`, đã được train trước trên `recipes.json` tại thời điểm khởi tạo hệ thống) để đưa ra quyết định phân loại nền ẩm thực. Mô hình ML là **người ra quyết định duy nhất** cho bước phân loại cuisine — không sử dụng IF-ELSE hay regex matching.

Các thành phần chi tiết:

- **NLP Parser (NER):** Ontology-based Entity Extraction trích xuất thực thể có tên từ văn bản tiếng Anh (nguyên liệu, ràng buộc y khoa, ngân sách).
- **TF-IDF Vectorization:** biến đổi tập công thức thành không gian vector đặc trưng, cho phép tính toán Cosine Similarity để xếp hạng mức độ phù hợp. Ma trận TF-IDF được lưu dưới dạng `data/tfidf_features.npy`.
- **Naive Bayes Classifier:** Multinomial Naive Bayes với Laplace smoothing, **learned from data** (train trên `recipes.json`). Tính toán Posterior $P(C|W) \propto P(C) \cdot \prod P(w_i|C)$ với log-sum-exp normalization để tránh underflow.

### 4.2. Ràng buộc (CSP) — L.O.1

Thuật toán thỏa mãn ràng buộc (Constraint Satisfaction Problem) với **Backtracking**, **Forward Checking nâng cao**, và heuristic **MRV** (Minimum Remaining Values).

**Kỹ thuật cốt lõi — Dual-constraint Pruning với Bounds Propagation:**

Forward Checking không chỉ kiểm tra ràng buộc ngân sách mà còn thực hiện **lan truyền cận trên/dưới cho biến Calo** (Bounds Propagation). Trước khi duyệt candidate cho mỗi biến chưa gán, hệ thống tính toán trước:

- $min\_future\_cals$: tổng calo **tối thiểu** từ Domain của tất cả các biến còn lại.
- $max\_future\_cals$: tổng calo **tối đa** từ Domain của tất cả các biến còn lại.

Điều kiện cắt tỉa kép (Dual-constraint Pruning):
```
if (current_cal + candidate_cal + min_future_cals) > max_cal:
    prune()  # Chắc chắn vi phạm cận trên
if (current_cal + candidate_cal + max_future_cals) < min_cal:
    prune()  # Chắc chắn không thể đạt đủ cận dưới
```

Kỹ thuật này chặn đứng các nhánh vi phạm ràng buộc calo ngay từ giai đoạn sớm ("chặn từ trong trứng nước"), thay vì chỉ phát hiện vi phạm khi đã gán xong tất cả biến.

### 4.3. Logic và tri thức (Knowledge Base) — L.O.2.1

Hệ chuyên gia dựa trên luật (Rule-based Expert System) sử dụng Propositional Logic. Tri thức được biểu diễn dưới dạng `Fact` (mệnh đề nguyên tử) và `PropositionalRule` (Horn Clause). Forward Chaining Engine suy diễn tiến để loại bỏ nguyên liệu vi phạm điều kiện y khoa (đau dạ dày, gout, tiểu đường, cao huyết áp, dị ứng lactose/gluten, bệnh thận, cholesterol cao).

### 4.4. Tìm kiếm (Search) — L.O.1

Thuật toán **A* (A-Star Search)** với **Admissible Heuristic** cho bài toán N-Days Meal Planning. Không gian trạng thái:

- **State:** `(day, total_cost, total_calories, sorted_multiset_of_recipes)`
- **Action:** chọn một công thức cho ngày tiếp theo.
- **Goal:** đủ N ngày, tổng chi phí ≤ budget, calo trong khoảng cho phép.
- **Cost:** tổng chi phí tích lũy $g(n)$; heuristic $h(n) = (N - day) \times min\_cost$ (Admissible vì không bao giờ đánh giá quá thực tế).

**Kỹ thuật cốt lõi — Canonical State Representation (Chuẩn hóa không gian trạng thái):**

Vì Goal Test chỉ kiểm tra **tổng chi phí** và **tổng calo** (không quan tâm thứ tự gán công thức cho ngày), các hoán vị (permutations) của cùng một tập công thức là **trạng thái tương đương**. Ví dụ: State(Phở, Pizza) ≡ State(Pizza, Phở).

Hệ thống áp dụng **Canonical Form** bằng cách sắp xếp `selected_recipes` trước khi nạp vào tập `visited`. Kỹ thuật này gộp các trạng thái hoán vị thành một, giảm kích thước cây tìm kiếm:

$$O(R^N) \rightarrow O\left(\frac{R^N}{N!}\right) = O(C(R,N))$$

Chuyển từ **Hoán vị** sang **Tổ hợp**, cho phép A* chạy thực tế với $N = 7$ ngày trở lên.

### 4.5. Xác suất (Bayesian Network) — L.O.2.2

Expert-driven Bayesian Network sử dụng Subjectivist Probabilities (xác suất gán bởi chuyên gia dinh dưỡng theo phương pháp Expert Elicitation — de Finetti). Mô hình gồm hai tầng inference:

1. **Posterior Preference Inference:** $P(Like | e_1, e_2, \ldots, e_n)$ tính qua Bayes' Rule trong log-space với log-sum-exp normalization.

2. **Extended Noisy-OR Model with Synergy Hidden Causes** (Mô hình Noisy-OR mở rộng với các nguyên nhân ẩn hiệp đồng):

   Mô hình cơ bản Noisy-OR giả định các yếu tố rủi ro là **độc lập**: $P(Risk) = 1 - \prod(1 - P_i)$. Tuy nhiên, trong thực tế dinh dưỡng, một số yếu tố có **tác động hiệp đồng** (synergy) khi xuất hiện đồng thời (ví dụ: Cay + Dầu mỡ gây kích ứng mạnh hơn tổng riêng lẻ).

   Thay vì nhân xác suất với hằng số (vi phạm tiên đề Kolmogorov), hệ thống mô hình hóa sự hiệp đồng như một **nguyên nhân ẩn (hidden cause)** bổ sung trong Noisy-OR:

   $$P(Risk) = 1 - \prod_{i}(1 - P_i) \times \prod_{j}(1 - P_{synergy_j})$$

   Trong đó $P_{synergy_j}$ là xác suất rủi ro độc lập của sự kiện hiệp đồng thứ $j$ (ví dụ: $P_{synergy}(Spicy \cap Oil) = 0.40$). Kết quả $P(Risk)$ luôn hội tụ trong $[0, 1]$ **một cách tự nhiên** theo lý thuyết xác suất, không cần hàm `min(0.95, ...)` để ép kiểu.

   *Tham khảo: Henrion (1989), "Some Practical Issues in Constructing Belief Networks", UAI'89.*

3. **Expected Utility Theory:** kết hợp Posterior Preference và Risk Assessment thành Expected Utility để ra quyết định tối ưu trong môi trường bất định.

---

## 5. Cấu trúc thư mục

```
yumify/
├── notebooks/            # Front-end Google Colab
│   └── main_frontend.ipynb   # Notebook chính — Runtime → Run All
├── modules/              # Các module AI cốt lõi
│   ├── __init__.py
│   ├── nlp_parser.py         # NLP Parser (NER + ML Naive Bayes tích hợp)
│   ├── ml_classifier.py      # Multinomial Naive Bayes Classifier
│   ├── search_engine.py      # A* Search (Canonical State Representation)
│   ├── csp_solver.py         # CSP Solver (Dual-constraint Pruning)
│   ├── knowledge_base.py     # Forward Chaining Engine (Propositional Logic)
│   └── bayes_risk.py         # Bayesian Network (Extended Noisy-OR + Synergy)
├── features/             # Trích xuất đặc trưng và kết nối dữ liệu
│   ├── __init__.py
│   ├── api_client.py         # Kết nối Spoonacular API
│   ├── data_loader.py        # Tải và parse dữ liệu JSON/CSV
│   ├── data_downloader.py    # Tải dữ liệu bổ sung
│   └── feature_extractor.py  # TF-IDF Vectorization + Heuristic Quantity
├── data/                 # Dữ liệu cục bộ
│   ├── recipes.json          # 25 công thức (6 nền ẩm thực)
│   ├── ingredients.json      # 70+ nguyên liệu (giá, calo, category)
│   ├── rules.json            # Luật suy diễn y khoa (Forward Chaining)
│   └── bayes_cpt.json        # CPT tables cho Bayesian Network
├── reports/              # Báo cáo
│   └── report.pdf            # Báo cáo PDF (EDA, pipeline, thí nghiệm)
└── README.md
```

---

## 6. Hướng dẫn cài đặt và vận hành

Dự án được thiết kế để chạy hoàn toàn trên **Google Colab**, tận dụng tài nguyên tính toán đám mây và tránh các vấn đề cấu hình môi trường cục bộ.

### Các bước thực hiện

1. Mở file `notebooks/main_frontend.ipynb` trên Google Colab.
2. Chạy **Runtime → Run all**.
3. Cell đầu tiên tự động clone mã nguồn từ GitHub (`https://github.com/dvydinh/yumify`) và cài đặt thư viện.
4. Cell tiếp theo tự động tải dataset Food.com từ Kaggle (`kagglehub`) vào môi trường Colab.
5. Nhập yêu cầu bằng tiếng Việt vào giao diện và nhấn nút phân tích.

Toàn bộ quá trình hoàn toàn tự động, không cần thao tác thủ công.

### Yêu cầu thư viện

```
underthesea
requests
ipywidgets
numpy
```

Tất cả các thư viện trên được cài đặt tự động trong notebook khi chạy trên Colab.

---

## 7. Liên kết

- Báo cáo PDF: `reports/report.pdf`
- Colab notebook: [`https://colab.research.google.com/drive/1y8RUaJrNEUVuj7jklSbMnOTUUEq5blHP?usp=sharing](https://colab.research.google.com/drive/1y8RUaJrNEUVuj7jklSbMnOTUUEq5blHP?usp=sharing)`
- GitHub repository: `https://github.com/dvydinh/yumify`

---

## 8. Tài liệu tham khảo

1. S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th Edition, Pearson, 2020. — Chương 3–4 (Search), Chương 6 (CSP), Chương 7 (Logic), Chương 13–16 (Probability & Bayes), Chương 20 (Naive Bayes).
2. P.E. Hart, N.J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 1968.
3. M. Henrion, "Some Practical Issues in Constructing Belief Networks", *Proceedings of the 3rd Conference on Uncertainty in AI (UAI)*, 1989. — Noisy-OR extensions and hidden causes.
4. G. Salton, C. Buckley, "Term-weighting approaches in automatic text retrieval", *Information Processing & Management*, 1988.
5. Spoonacular API Documentation: https://spoonacular.com/food-api/docs
