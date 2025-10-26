# 🚀 RAG vs Full Context AI Benchmark

## 📋 Overview

This project compares **RAG (Retrieval-Augmented Generation)** and **Full Context** approaches for AI response generation, using OpenAI GPT-4o-mini and Google Gemini 2.0-flash models.

**Based on Clearservice company data** - A comprehensive comparison of document retrieval vs full context methods for corporate knowledge bases.

## 🛠️ Setup & Installation

### 1. **Create Python Virtual Environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. **Install Requirements**

```bash
#Install requirements
pip install -r requirements.txt
```

### 3. **Environment Configuration**

Create a `.env` file in the project root with your API keys:

```env
# .env file example
OPENAI_API_KEY=sk-your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

**How to get API keys:**

- **OpenAI:** Visit [platform.openai.com](https://platform.openai.com/api-keys)
- **Gemini:** Visit [makersuite.google.com](https://makersuite.google.com/app/apikey)

## ✅ Technical Configuration

### 🤖 AI Models:

- **OpenAI:** `gpt-4o-mini` - stable, fast, more expensive
- **Gemini:** `gemini-2.0-flash` - optimized prompts, rate limited, cheaper

### 🔍 RAG Components:

- **Vector Database:** ChromaDB with cosine similarity
- **Embedding Model:** `paraphrase-multilingual-MiniLM-L12-v2` via SentenceTransformers
- **Retrieval:** Top-3 document chunks
- **Chunk Strategy:** Section-based splitting (300 char max)

### 📊 Testing Methods:

- **RAG:** Top-3 document retrieval with ChromaDB
- **FullContext:** Complete document context

## 🚀 Usage

### **Run Unified Benchmark**

```bash
python benchmark.py
```

**Available Options:**

1. **Quick test (3 questions)**
2. **Medium test (10 questions)**
3. **FULL TEST (50 questions)**

### **Visualization**

```bash
python visualize_simple.py
```

**Outputs:** 2 PNG charts (6 core metrics + topic performance)

## ⚡ Rate Limiting

**Gemini Free Tier limits:**

- **8 requests/minute** (strict stability)
- **1400 requests/day** (daily limit monitoring)
- Automatic waiting and progress tracking

## 🔧 Gemini Optimizations

### ✅ Working prompt template:

```
Answer user question based on document.

Document content:
{context}

User question:
{question}

Please provide a short, fact-based answer based on the document information:
```

### 🛡️ Safe word replacements:

- `Clearservice` → `organization`
- `company` → `organization`
- `Hungarian employee` → `employee`

### ⚙️ Safety settings:

```python
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
]
```

## 📁 Project Structure

### 🔧 Core Components:

- `benchmark.py` - **Unified benchmark script** (RAG + AI testing)
- `rag_system.py` - ChromaDB and embeddings
- `visualize_simple.py` - **Simplified visualization** (7 charts)

### 📊 Data:

- `data/cs_qa.csv` - 50 question-answer pairs (Clearservice knowledge base)
- `data/topics.txt` - Full context document (Clearservice company information)

### 📈 Results (in docs/ folder):

- `docs/benchmark_results_YYYYMMDD_HHMM.csv` - Final benchmark results
- `docs/temp_results_X.csv` - Intermediate saves
- `docs/main_comparison_optimized_results.png` - 6 core charts
- `docs/topic_performance_optimized_results.png` - Topic performance

## 🎯 Result Metrics

### 📊 Quality Metrics:

- **word_overlap:** Word overlap score (0-1)
- **contains_key_info:** Key information coverage (0-1)
- **exact_match:** Exact answer match (0-1)

### ⏱️ Performance Metrics:

- **generation_time:** Response generation time (seconds)
- **tokens_used:** Tokens consumed
- **cost_estimate:** Estimated cost ($)

## 📊 Visualization

### 🎨 Simplified Charts (visualize_simple.py):

1. **Average response quality** (Word Overlap)
2. **Average response time** (seconds)
3. **Total estimated cost** ($)
4. **Context size vs quality**
5. **Token usage distribution**
6. **Final combined score** (weighted)
7. **Topic performance heatmap**

### 📁 Output Location:

- All results (CSV + PNG) in **docs/** folder

## 🔍 Previous Results

### 🏆 OpenAI dominance:

- **Quality:** OpenAI consistently better
- **Speed:** Similar (~1-2s)
- **Cost:** OpenAI ~10x more expensive

### 📊 RAG vs Full Context:

- **RAG:** More efficient, focused
- **Full Context:** Slightly better quality, but more expensive

## ⚠️ Troubleshooting

### 🚫 Gemini "cannot generate for safety reasons":

1. Check model name: `gemini-2.0-flash` ✅
2. Use optimized prompt template
3. Try new Google account and API key

### 🐌 Slow execution:

- Normal with Gemini Free Tier (8 requests/minute)
- 50 questions
- Progress tracking every 5 questions

## 📋 Requirements

This project requires the following Python packages:

**Core dependencies** (install via `pip install -r requirements.txt`):

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
openai>=1.0.0
google-generativeai>=0.3.0
python-dotenv>=1.0.0
```
## � Data Source

This benchmark is based on **Clearservice company data**, including:

- Company policies and procedures
- Employee handbook information
- IT support documentation
- HR processes and guidelines
- Administrative procedures

The dataset contains 50 carefully curated question-answer pairs across 12 different topic categories, providing comprehensive coverage of corporate knowledge base scenarios.

---

**Last Updated:** 2025-10-26  
