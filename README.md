# Chatbot PDF RAG

Chatbot RAG su dung Streamlit + LangChain + Google Gemini + FAISS local.

Project nay cho phep:

- Upload nhieu file PDF
- Tach text thanh chunks va tao embeddings
- Luu vector database local bang FAISS trong thu muc `vectorstores/faiss`
- Hoi dap theo noi dung tai lieu
- Download lich su hoi dap ra CSV

## 1. Yeu cau

- Python 3.10+
- Google AI API key (Gemini)

## 2. Cai dat

### Cach 1: dung uv

```bash
uv sync
```

### Cach 2: dung pip + venv

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## 3. Cau hinh API key

Tao file `.env` trong root project:

```env
GOOGLE_API_KEY=your_api_key_here
```

Ban cung co the nhap key truc tiep trong sidebar cua app.

## 4. Chay ung dung

```bash
streamlit run app.py
```

Mo URL Streamlit hien thi trong terminal, thuong la:

```text
http://localhost:8501
```

## 5. Quy trinh su dung

1. Nhap Google API key
2. Upload 1 hoac nhieu file PDF
3. Bam **Submit and Process** de tao index FAISS
4. Dat cau hoi trong chat box
5. Download lich su hoi dap (CSV) neu can

## 6. Cau truc project

```text
chatbot-pdf-rag/
├─ app.py
├─ pyproject.toml
├─ README.md
└─ vectorstores/
   └─ faiss/
      └─ current_index/    # tao sau khi bam Submit and Process
```

## 7. Diem da hoan thien

- Fix import LangChain theo API hien tai
- Tach ro 2 buoc Process PDF va Chat truy van
- Luu FAISS local vao thu muc rieng trong project
- Quan ly session state on dinh (chat history, index status)
- Xu ly loi co ban khi PDF khong co text hoac key bi thieu

## 8. Luu y

- PDF scan anh co the khong trich xuat duoc text voi PyPDF2.
- App hien dang su dung 1 model: Google AI (Gemini).
- FAISS index local phu hop POC va du an nho. Neu muon scale production cloud, co the nang cap sang Pinecone/Qdrant.

## 9. Huong nang cap tiep

- Them OCR cho scanned PDF (pytesseract, unstructured)
- Them metadata retrieval (page number, source file)
- Them citation tung doan context trong cau tra loi
- Them test tu dong cho text extraction va retrieval flow
