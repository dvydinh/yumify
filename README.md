# Hệ thống AI gợi ý ẩm thực và tối ưu hóa ngân sách

**Bài tập lớn môn học:** Nhập môn Trí tuệ Nhân tạo (Introduction to AI)
**Mã môn học:** CO3061
**Học kỳ:** II — Năm học 2025–2026
**Trường:** Đại học Bách Khoa, ĐHQG-HCM
**Giảng viên hướng dẫn:** TS. Trương Vĩnh Lân
**GitHub repository:** [https://github.com/dvydinh/yumify](https://github.com/dvydinh/yumify)

---

## 1. Giới thiệu tổng quan

Dự án xây dựng một hệ thống Trí tuệ Nhân tạo tích hợp (Hybrid AI System), kết hợp AI cổ điển (Symbolic AI, Search) và Học máy (Machine Learning, Generative AI) để giải quyết bài toán gợi ý thực đơn đa ràng buộc. Hệ thống tiếp nhận yêu cầu bằng ngôn ngữ tự nhiên tiếng Anh, phân tích ngữ nghĩa, và đề xuất công thức nấu ăn thỏa mãn đồng thời các ràng buộc về chi phí, lượng calo, và điều kiện y khoa của người dùng.

Hệ thống bao phủ 6 nền ẩm thực: Việt Nam, Ý, Nhật Bản, Hàn Quốc, Mexico, và Phương Tây.

---

## 2. Thông tin nhóm

| Họ và tên | MSSV | Email |
|-----------|------|-------|
| Đinh Đoàn Vy | 2353350 | vy.dinhdoan2005@hcmut.edu.vn |
| Trần Thiên Lộc | 2352715 |  |
| Nguyễn Trần Gia Bảo | 2352103 |  |
| Dương Lê Nhật Duy | 2352171 |  |

---

## 3. Nguồn dữ liệu

### 3.1. Dataset chính

Hệ thống sử dụng tập dữ liệu **"Food.com Recipes and User Interactions"** từ Kaggle (được công bố bởi tác giả `shuyangli94`). Đây là tập dữ liệu quy mô lớn với hơn 230,000 công thức, được sử dụng rộng rãi trong các nghiên cứu về Recommender Systems. Hệ thống tự động fetch, parse, và trích xuất đặc trưng TF-IDF từ tập dữ liệu này khi chạy trên Colab.

### 3.2. APIs bổ trợ

- **Spoonacular API:** cung cấp fallback data thời gian thực, bao gồm thông tin dinh dưỡng và giá cả nguyên liệu khi dataset cục bộ không đáp ứng truy vấn.
- **Hugging Face Inference API (Mistral-7B và SDXL):** kích hoạt cơ chế Generative Fallback để tổng hợp công thức mới khi không gian trạng thái (State Space) của database không thỏa mãn ràng buộc người dùng.

---

## 4. Kiến trúc hệ thống và các trụ cột AI

Pipeline xử lý tổng thể:

```
User Input → NLP Parser → TF-IDF Vectorization → CSP Solver → A* Search → Knowledge Base → Bayesian Risk → Output
```

Hệ thống được xây dựng trên 5 trụ cột AI theo yêu cầu đề bài:

### 4.1. Học máy (Machine Learning) — L.O.3

- **NLP Parser (NER):** sử dụng kiến trúc Ontology và Regex để trích xuất thực thể có tên (Named Entity Recognition) từ văn bản tiếng Anh, nhận diện nguyên liệu, ẩm thực mục tiêu, và các ràng buộc y khoa ngầm định.
- **TF-IDF Vectorization:** biến đổi tập công thức thành không gian vector đặc trưng, cho phép tính toán Cosine Similarity để xếp hạng mức độ phù hợp giữa truy vấn và công thức. Ma trận TF-IDF được lưu dưới dạng `data/tfidf_features.npy`.
- **Naive Bayes Classifier:** phân loại ẩm thực (Cuisine Classification) dựa trên tập nguyên liệu đầu vào, sử dụng Multinomial Naive Bayes với Laplace smoothing. Đây là thành phần ML duy nhất có xác suất **learned from data**.

### 4.2. Ràng buộc (CSP) — L.O.1

Thuật toán thỏa mãn ràng buộc (Constraint Satisfaction Problem) với Backtracking kết hợp Forward Checking và heuristic MRV (Minimum Remaining Values) để thu hẹp không gian tìm kiếm dựa trên giới hạn calo tối đa và ngân sách cho phép.

### 4.3. Logic và tri thức (Knowledge Base) — L.O.2.1

Hệ chuyên gia dựa trên luật (Rule-based Expert System) sử dụng Propositional Logic. Tri thức được biểu diễn dưới dạng `Fact` (mệnh đề nguyên tử) và `PropositionalRule` (Horn Clause). Forward Chaining Engine suy diễn tiến để loại bỏ nguyên liệu vi phạm điều kiện y khoa (đau dạ dày, gout, tiểu đường, cao huyết áp, dị ứng lactose/gluten, bệnh thận, cholesterol cao).

### 4.4. Tìm kiếm (Search) — L.O.1

Thuật toán A* (A-Star Search) với Admissible Heuristic để tìm lộ trình tối ưu cho bài toán giỏ hàng mua sắm. Không gian trạng thái được mô hình hóa với:

- **State:** tập nguyên liệu đã chọn và tổng chi phí hiện tại.
- **Action:** thêm một nguyên liệu vào giỏ hàng.
- **Goal:** giỏ hàng đủ nguyên liệu cho công thức được chọn.
- **Cost:** tổng chi phí bằng VNĐ.

### 4.5. Xác suất (Bayesian Network) — L.O.2.2

Expert-driven Bayesian Network sử dụng Subjectivist Probabilities (xác suất gán bởi chuyên gia dinh dưỡng, không phải learned from data). Tính toán Posterior Preference P(Like|Evidence) qua Bayes' Rule trong log-space, đánh giá rủi ro tiêu hóa qua Independent Risk Model (Noisy-OR), và kết hợp thành Expected Utility Theory để ra quyết định trong môi trường bất định.

---

## 5. Cấu trúc thư mục

```
NMAI/
├── modules/              # Các module AI cốt lõi
│   ├── nlp_parser.py         # NLP Parser (Ontology, Regex NER tiếng Anh)
│   ├── search_engine.py      # A* Search tối ưu giỏ hàng
│   ├── csp_solver.py         # CSP Solver (Backtracking + MRV)
│   ├── knowledge_base.py     # Forward Chaining Engine (Propositional Logic)
│   ├── bayes_risk.py         # Expert-driven Bayesian Network + Expected Utility
│   └── ml_classifier.py      # Naive Bayes Cuisine Classifier (ML)
├── features/             # Trích xuất đặc trưng và kết nối dữ liệu
│   ├── api_client.py         # Kết nối Spoonacular, HuggingFace
│   ├── data_loader.py        # Tải và parse dữ liệu JSON/CSV
│   ├── data_downloader.py    # Tải dữ liệu từ Spoonacular API
│   └── feature_extractor.py  # TF-IDF Vectorization và lưu .npy
├── notebooks/            # Giao diện chạy trên Google Colab
│   └── main_frontend.ipynb   # Notebook chính (front-end)
├── data/                 # Dữ liệu nguyên liệu, công thức, và TF-IDF
│   ├── ingredients.json
│   ├── recipes.json
│   └── tfidf_features.npy
├── reports/              # Báo cáo LaTeX
│   └── report.tex
└── tests/                # Unit tests
    └── test_modules.py
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
- Colab notebook: `https://colab.research.google.com/drive/1OfIx35btL-6WLyd6XocZ9pXxWFP53a9b?usp=sharing`
- GitHub repository: `https://github.com/dvydinh/yumify`

---

## 8. Tài liệu tham khảo

1. S. Russell, P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th Edition, Pearson, 2020.
2. P.E. Hart, N.J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 1968.
3. G. Salton, C. Buckley, "Term-weighting approaches in automatic text retrieval", *Information Processing & Management*, 1988.
4. Spoonacular API Documentation: https://spoonacular.com/food-api/docs
5. Hugging Face Inference API: https://huggingface.co/docs/api-inference
