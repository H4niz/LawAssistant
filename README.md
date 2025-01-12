# 📚 Dự Án Nghiên Cứu Xử Lý Văn Bản Pháp Lý

### Giới thiệu
Dự án này sử dụng các mô hình học sâu (PhoBERT và VinaLLama) để xử lý, tìm kiếm và trả lời các câu hỏi liên quan đến văn bản pháp luật.

---

## 🚀 Hướng dẫn triển khai

### 1️⃣ Yêu cầu hệ thống
- Python 3.8.10
- GPU hỗ trợ CUDA (tùy chọn, nhưng khuyến nghị)

### 2️⃣ Cài đặt
1. **Clone dự án:**
   ```bash
   git clone https://github.com/username/research-project.git
   cd research-project

### Cấu trúc dự án

```
research-project/
│
├── src/
│   ├── main.py                  # File chính chứa logic của hệ thống
│   ├── models/
│   │   ├── model_manager.py     # Quản lý các mô hình PhoBERT và VinaLLama
│   │   └── model_config.py      # Định nghĩa cấu hình cho mô hình
│   ├── utils/
│   │   ├── indexing.py          # Chức năng tạo và tải Annoy Index
│   │   └── text_processing.py   # Các hàm xử lý đoạn văn bản và truy vấn
│   └── inference/
│       ├── embedding.py         # Hàm tạo vector nhúng từ văn bản
│       └── answer_generation.py # Hàm sinh câu trả lời từ query và context
│
├── data/
│   ├── legal_docs.txt           # Các tài liệu luật pháp (dataset mẫu)
│   ├── legal_docs.ann           # Annoy Index lưu trữ embeddings
│   └── sample_queries.txt       # Các truy vấn mẫu
│
├── logs/
│   └── system.log               # File log để theo dõi hệ thống
│
├── requirements.txt             # Danh sách thư viện Python cần cài đặt
├── README.md                    # Hướng dẫn sử dụng và triển khai dự án
└── LICENSE                      # Thông tin giấy phép
└── docs                         # Tài liệu


```