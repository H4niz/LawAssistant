## **Dự án: Hệ thống hỗ trợ pháp lý dựa trên học sâu**

### **Tóm tắt dự án**
Dự án này tập trung vào việc xây dựng hệ thống xử lý và truy vấn văn bản pháp lý bằng cách sử dụng các mô hình học sâu như PhoBERT và VinaLLama. Hệ thống tích hợp công cụ truy vấn ngữ nghĩa với cơ chế sinh văn bản nhằm hỗ trợ người dùng trả lời các câu hỏi pháp lý một cách chính xác và hiệu quả.

---

### **Giải thích mô hình RAG**

Mô hình **RAG (Retrieval-Augmented Generation)** kết hợp hai thành phần chính:  
1. **Retriever (Bộ tìm kiếm):**  
   - Sử dụng Annoy để tìm kiếm các tài liệu liên quan từ cơ sở dữ liệu lớn dựa trên vector nhúng (embedding).  
   - Quá trình này đảm bảo rằng các tài liệu phù hợp nhất với truy vấn người dùng được chọn làm ngữ cảnh.  

2. **Generator (Bộ sinh):**  
   - Sử dụng mô hình sinh ngôn ngữ (VinaLLama) để tạo câu trả lời dựa trên ngữ cảnh đã truy xuất.  
   - Câu trả lời được định dạng rõ ràng, kèm theo nguồn tham chiếu từ các văn bản pháp luật.  

#### **Lợi ích của RAG trong hệ thống**  
- **Hiệu quả:** Tích hợp truy xuất thông tin và sinh câu trả lời nhanh chóng.  
- **Chính xác:** Trả lời dựa trên ngữ cảnh cụ thể, hạn chế thông tin sai lệch.  
- **Tùy chỉnh:** Dễ dàng mở rộng cơ sở dữ liệu hoặc điều chỉnh mô hình cho các lĩnh vực khác nhau.  

---

### **Mô hình hoạt động**

Dưới đây là sơ đồ minh họa quy trình hoạt động của hệ thống RAG:

![RAG Workflow](A_clear_and_professional_diagram_illustrating_the_.png)

1. **Người dùng nhập truy vấn:** Hệ thống nhận đầu vào là câu hỏi từ người dùng.  
2. **Truy xuất tài liệu:** Sử dụng Annoy Index để tìm các tài liệu pháp lý liên quan dựa trên vector nhúng.  
3. **Lấy ngữ cảnh:** Ghép các đoạn văn bản phù hợp để tạo ngữ cảnh cho mô hình sinh.  
4. **Sinh câu trả lời:** Mô hình VinaLLama xử lý truy vấn và ngữ cảnh để tạo ra câu trả lời rõ ràng, chính xác.  
5. **Trả về kết quả:** Câu trả lời được gửi lại cho người dùng cùng với nguồn tham chiếu.  

---

### **Cài đặt và triển khai**

#### **Yêu cầu hệ thống**
- Python 3.8.10  
- GPU có hỗ trợ CUDA (khuyến khích để tăng tốc xử lý).  

#### **Cài đặt thư viện**
Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

#### **Chạy dự án**
1. Đảm bảo rằng bạn đã chuẩn bị sẵn dữ liệu và mô hình.
2. Khởi động ứng dụng bằng lệnh:
```bash
python src/main.py
```
3. Nhập câu hỏi và nhận câu trả lời từ hệ thống.

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


---

### **Liên hệ**
- **Tác giả:** Nguyễn Lê Quốc Anh và Tô Duy Hinh  
- Mục đích: Chia sẻ mã nguồn cho học tập và nghiên cứu.