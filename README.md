# RAG-Powered Complaint Analysis Chatbot

## Overview
An intelligent complaint analysis system for CrediTrust Financial that transforms unstructured customer feedback into actionable insights using Retrieval-Augmented Generation (RAG). The system enables product managers, support teams, and compliance officers to query complaint data in plain English and receive evidence-backed answers in seconds.


## Data Pipeline

###  EDA & Preprocessing
- **Input**: Full CFPB dataset (~9.6M records, 5.63GB)
- **Filtered**: 885,813 records for 4 products (Credit Cards, Personal Loans, Savings Accounts, Money Transfers)
- **Cleaned**: 435,737 records with narratives
- **Output**: `data/processed/filtered_complaints.csv`

###  Chunking & Embedding
- **Sampling**: 12,000 stratified records (preserves product distribution)
- **Chunking**: 500 characters with 50-character overlap
- **Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **Vector Store**: FAISS index
- **Output**: `vector_store/` (faiss_index.bin, chunks.pkl, metadatas.pkl)

## Key Findings

### Product Distribution
| Category | Total | With Narratives | Coverage |
|----------|-------|-----------------|----------|
| Credit Cards | 433,055 | 189,334 | 43.7% |
| Savings Accounts | 291,178 | 140,319 | 48.2% |
| Money Transfers | 145,066 | 97,188 | 67.0% |
| Personal Loans | 16,514 | 8,896 | 53.9% |

### Narrative Length
- **Average**: 206 words (~1,200 characters)
- **Median**: 137 words (~800 characters)
- **Range**: 1 - 6,472 words

### Chunk Size Justification
- **Selected**: 500 characters with 50 overlap
- **Avg chunks per doc**: 2.85 (ideal: 1.5-2.5)
- **Avg chunk length**: 369 characters (ideal: 300-600)
- **Result**: 36,512 chunks from 12,000 complaints

## Project Structure
```
rag-complaint-chatbot/
├── data/
│ ├── raw/ # Raw CFPB data
│ └── processed/ # Cleaned datasets
├── notebooks/
│ ├── 01_eda_preprocessing.ipynb
│ └── 02_chunking.ipynb
├── src/
│ ├── data_processor.py # EDA & preprocessing logic
│ └── chunker.py # Chunking & embedding logic
├── vector_store/ # FAISS index & artifacts
├── tests/ # Unit tests
├── requirements.txt
└── README.md

```


## Setup Instructions

### Prerequisites
- Python 3.11+
- 8GB+ RAM recommended

### 1. Clone Repository
```bash
git clone https://github.com/meronsisay/rag-complaint-chatbot.git
cd rag-complaint-chatbot

python -m venv venv
source venv/bin/activate          # Mac/Linux
# or
venv\Scripts\activate              # Windows

pip install -r requirements.txt

# run eda and preprocessing
jupyter notebook notebooks/eda.ipynb

# build vector store
python src/chunker.py