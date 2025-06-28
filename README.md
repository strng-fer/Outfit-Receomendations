# Sistem Rekomendasi Outfit 👗✨

Sistem rekomendasi outfit berbasis AI yang menggunakan teknologi RAG (Retrieval-Augmented Generation) dengan NVIDIA LLM dan FAISS vector store untuk memberikan rekomendasi outfit yang personal dan kontekstual.

## 📋 Deskripsi Proyek

Program ini adalah sistem AI yang dapat memberikan rekomendasi outfit sesuai dengan preferensi pengguna, jenis acara, cuaca, dan faktor lainnya. Sistem ini memanfaatkan:

- **RAG Pipeline**: Menggabungkan pencarian dokumen relevan dengan generasi teks
- **NVIDIA LLM**: Menggunakan model bahasa besar dari NVIDIA AI Endpoints
- **FAISS Vector Store**: Untuk pencarian semantik yang cepat dan efisien
- **HuggingFace Embeddings**: Untuk mengkonversi teks menjadi representasi vektor


Proyek bertujuan untuk menyelesaikan kelas bootcamp Generative AI with LLM  yang diadakanoleh Navasena. 

### Certificate
#### 🎓 Sertifikat Penyelesaian Kelas
![Certificate of Participation](./Feryadi%20Yulius-Certificate%20of%20Participation.png)

#### 🏆 Sertifikat Final Project
![Certificate of Final Project Completion](./Feryadi%20Yulius-Final%20Project%20Completion.png)

## 🎯 Fitur Utama

- ✅ Rekomendasi outfit berdasarkan acara dan cuaca
- ✅ Pencarian semantik pada database outfit
- ✅ Interface interaktif untuk tanya jawab
- ✅ Prompt engineering untuk respons yang lebih natural
- ✅ Penyimpanan dan loading vector store

## 🛠️ Teknologi yang Digunakan

- **Python 3.8+**
- **LangChain**: Framework untuk aplikasi LLM
- **NVIDIA AI Endpoints**: Model bahasa besar
- **FAISS**: Vector database untuk similarity search
- **HuggingFace Transformers**: Pre-trained embeddings
- **Pandas**: Data manipulation
- **Jupyter Notebook**: Development environment

## 📦 Instalasi dan Setup

### 1. Clone Repository
```bash
git clone https://github.com/username/Outfit-Receomendations.git
cd Outfit-Receomendations
```

### 2. Install Dependencies
```bash
# Install system dependencies
apt-get install sox libsndfile1 ffmpeg

# Install Python packages
pip install uv
uv pip install langchain langgraph langchain_nvidia_ai_endpoints
uv pip install langchain-community
uv pip install pandas sentence-transformers faiss-cpu
uv pip install jupyter ipython
```

### 3. NVIDIA API Key Setup
Dapatkan API key dari [NVIDIA AI Foundation](https://build.nvidia.com/) dan set sebagai environment variable:
```bash
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### 4. Jalankan Notebook
```bash
jupyter notebook "NAVASENA (1).ipynb"
```

## 🚀 Cara Penggunaan

### 1. Persiapan Data
- Dataset outfit akan dimuat otomatis dari HuggingFace Hub
- Data akan diproses dan dikonversi ke format JSONL
- Vector embeddings akan dibuat untuk pencarian semantik

### 2. Membangun Vector Store
```python
# Load dataset
df = pd.read_csv("hf://datasets/formido/outfit_recomendation/new_data.csv")

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Build FAISS vector store
vectorstore = FAISS.from_texts(texts, embeddings)
```

### 3. Menggunakan Sistem Rekomendasi
```python
# Setup retrieval chain
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Tanya jawab interaktif
result = qa_chain.run("Berikan rekomendasi outfit untuk acara formal dengan cuaca dingin")
```

## 💡 Contoh Query

- "Rekomendasikan outfit untuk meeting bisnis di cuaca panas"
- "Apa yang harus dipakai untuk acara dinner formal?"
- "Outfit casual untuk hangout dengan teman di weekend"
- "Pakaian yang cocok untuk traveling ke pantai"

## 📁 Struktur Proyek

```
Outfit-Receomendations/
├── NAVASENA (1).ipynb          # Main notebook
├── README.md                   # Project documentation
├── Final Project Brief.pdf     # Project brief
├── PPT_FINAL PROJECT_FERYADIYULIUS.pdf  # Presentation
└── faiss_outfit_recommendation_index/   # Saved vector store (generated)
```

## 🔧 Konfigurasi

### Environment Variables
```bash
NVIDIA_API_KEY=your_nvidia_api_key
```

### Model Configuration
- **LLM Model**: NVIDIA AI Endpoints
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS with similarity search
- **Retrieval**: Top-k=3 similar documents

## 📊 Dataset

Dataset menggunakan [Outfit Recommendation Dataset](https://huggingface.co/datasets/formido/outfit_recomendation) dari HuggingFace Hub yang berisi:
- Instruksi outfit
- Input konteks (acara, cuaca, preferensi)
- Output rekomendasi

## 🤝 Kontributor

- **Feryadi Yulius** - Undergraduated Data Science Student
- **Kelas**: Generative AI with LLM

## 📄 Lisensi

Project ini dibuat untuk keperluan edukasi dan pembelajaran AI.

## 🔗 Links

- [HuggingFace Dataset](https://huggingface.co/datasets/formido/outfit_recomendation)
- [NVIDIA AI Endpoints](https://build.nvidia.com/)
- [LangChain Documentation](https://docs.langchain.com/)

## ⚠️ Catatan Penting

1. **GPU Requirement**: Disarankan menggunakan Google Colab dengan GPU untuk performa optimal
2. **API Key**: Pastikan NVIDIA API key valid dan tidak di-hardcode dalam kode
3. **Dependencies**: Beberapa package memerlukan versi khusus (numpy==1.26.4)

## 🎨 Preview

Sistem ini memberikan rekomendasi outfit yang detail dengan penjelasan mengapa setiap item cocok untuk situasi tertentu, menciptakan pengalaman seperti berkonsultasi dengan fashion stylist profesional.

---

*Dibuat dengan ❤️ menggunakan AI dan teknologi terdepan untuk fashion recommendation*
