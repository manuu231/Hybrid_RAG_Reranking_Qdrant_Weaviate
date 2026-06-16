<div align="center">

# 🔍 Advanced RAG — Hybrid Retrieval + Reranking + Qdrant + Weaviate

### Beyond Basic RAG — Production Grade Retrieval Systems
*Hybrid Search + BM25 + Reranking + Qdrant Vector Database*

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=for-the-badge&logo=chainlink&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?style=for-the-badge)
![Weaviate](https://img.shields.io/badge/Weaviate-Hybrid_Search-blue?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Facebook_AI-orange?style=for-the-badge&logo=meta&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=for-the-badge&logo=huggingface&logoColor=white)

<br/>

**Built by [Manpreet Kaur](https://github.com/manuu231)**
MS Data Science @ Clarkson University | AI/ML Engineer | 3+ Years @ Wipro

</div>

---

## 📌 What Does This Project Do?

Goes beyond basic RAG to implement production-grade retrieval techniques that deliver more accurate and relevant results!

```
Basic RAG:
Question → Semantic Search → Top 3 chunks → LLM → Answer
(sometimes wrong results!)

Advanced RAG (this project):
Question → Hybrid Search + Reranking → Best chunks → LLM → Answer
(much more accurate!) ✅
```

---

## 🧠 4 Retrieval Techniques Compared

### 1️⃣ Semantic Search
```
Searches by MEANING
"fastest database" → finds "quickest vector store"
Great for natural language queries!
```

### 2️⃣ Keyword Search (BM25)
```
Searches by EXACT WORDS
"fastest database" → finds documents with exact words
Fast but misses synonyms and context!
```

### 3️⃣ Hybrid Search ⭐
```
Combines SEMANTIC + KEYWORD
Best of both worlds!
Semantic: 70% weight
BM25:     30% weight
→ More accurate results!
```

### 4️⃣ Reranking
```
Retrieve MORE → Score again → Keep BEST
Step 1: Get top 10 results
Step 2: Score each for relevance
Step 3: Keep only top 3
→ Even more precise results!
```

---

## 📊 Results Comparison

Query: **"Which database is fastest?"**

| Method | Result 1 | Result 2 | Result 3 | Accuracy |
|--------|---------|---------|---------|---------|
| Semantic | ✅ Qdrant | ✅ FAISS | ✅ Weaviate | Good |
| BM25 Keyword | ❌ RAG | ✅ Qdrant | ✅ FAISS | Poor |
| **Hybrid** | ✅ **Qdrant** | ✅ FAISS | ✅ Pinecone | **Better** |
| **Reranking** | ✅ **Qdrant** | ✅ FAISS | ✅ Pinecone | **Best!** |

> Hybrid and Reranking correctly identified Qdrant as the fastest database!
> BM25 alone matched "which" keyword incorrectly returning RAG document!

---

## 🗄️ Vector Databases Covered

### Qdrant
```python
# Create client
client = QdrantClient(":memory:")

# Create collection
client.create_collection(
    collection_name="ai_knowledge",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Search
results = client.query_points(
    collection_name="ai_knowledge",
    query=query_vector,
    limit=3
).points
```

**Why Qdrant?**
- Built with Rust → fastest vector search ⚡
- Runs locally OR cloud
- Free tier available
- Best for high performance production apps

---

### Weaviate
```python
# Hybrid search with alpha control
result = client.query.get("Document", ["text"])
    .with_hybrid(
        query="fastest database",
        alpha=0.5  # 50% semantic + 50% keyword
    )
    .with_limit(3).do()
```

**Why Weaviate?**
- Built-in hybrid search
- GraphQL API for flexible queries
- `alpha` controls semantic vs keyword balance
- Great for complex query requirements

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python | Programming language |
| 🔗 LangChain | AI framework |
| 🗄️ FAISS | Local vector database |
| 📊 BM25 (rank-bm25) | Keyword search algorithm |
| 🔴 Qdrant | Fast Rust-based vector database |
| 🔵 Weaviate | Hybrid search vector database |
| 🤗 sentence-transformers | Local embeddings (384 dimensions) |

---

## 🔄 How Hybrid Search Works

```
Query: "fastest database"
         │
         ├──► Semantic Search (FAISS)
         │    Converts query to vector
         │    Finds similar meaning documents
         │    Returns: Qdrant(0.83), FAISS(0.69)
         │
         ├──► BM25 Keyword Search
         │    Tokenizes query words
         │    Finds exact keyword matches
         │    Returns: RAG(1.73), Qdrant(0.77)
         │
         └──► Combine Scores
              Semantic score + BM25 score * 0.3
              Remove duplicates
              Sort by combined score
              
Final: Qdrant(1.06) → FAISS(0.91) → Pinecone(0.65) ✅
```

---

## 🔄 How Reranking Works

```
Query → Retrieve TOP 10 (cast wide net)
              ↓
        Score each document
        "How many query words match?"
              ↓
        Remove irrelevant documents
        (score = 0.00 → removed!)
              ↓
        Keep TOP 3 highest scored
              ↓
        Send to LLM → Better answer! ✅
```

---

## ⚙️ Setup and Installation

### Step 1 — Clone Repository
```bash
git clone https://github.com/manuu231/Hybrid_RAG_Reranking_Qdrant_Weaviate.git
cd Hybrid_RAG_Reranking_Qdrant_Weaviate
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Open in Google Colab
Upload `day12_hybrid_retrieval_reranking.ipynb` to Colab and run all cells!

> No API key needed for embeddings — sentence-transformers runs locally!

---

## 🔑 Key Concepts

<details>
<summary><b>What is BM25?</b></summary>

BM25 (Best Match 25) is a keyword-based ranking algorithm used by search engines like Elasticsearch. It scores documents based on how many query words appear in each document, considering word frequency and document length.

```python
from rank_bm25 import BM25Okapi

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
scores = bm25.get_scores(query.lower().split())
```
</details>

<details>
<summary><b>What is Hybrid Search alpha?</b></summary>

Alpha controls the balance between semantic and keyword search:
- `alpha=0.0` → Pure keyword search
- `alpha=0.5` → 50% semantic + 50% keyword
- `alpha=1.0` → Pure semantic search
- `alpha=0.7` → Recommended for most use cases
</details>

<details>
<summary><b>Why Reranking?</b></summary>

Initial retrieval casts a wide net (top 10). Reranking applies a more sophisticated scoring model to these candidates to find the truly most relevant ones. This two-stage approach balances speed (fast initial retrieval) with accuracy (careful reranking).
</details>

---

## 📁 Project Structure

```
Hybrid_RAG_Reranking_Qdrant_Weaviate/
│
├── hybrid_retrieval_reranking.ipynb  ← Main notebook
├── requirements.txt                         ← Dependencies
└── README.md                                ← This file
```

---

## 📄 License
MIT License — free to use and modify!

---

<div align="center">

⭐ **Star this repo if you found it helpful!**

Made with ❤️ by [Manpreet Kaur](https://github.com/manuu231)

🔗 [GitHub](https://github.com/manuu231) | 
🤗 [Hugging Face](https://huggingface.co/Manpreet02)

</div>
