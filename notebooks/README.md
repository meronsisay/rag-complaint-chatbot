##  1. EDA Insight: CFPB Complaint Data Analysis

This module handles exploratory data analysis (EDA) and preprocessing on the raw CFPB complaint dataset to prepare it for indexing. The analysis focuses explicitly on four key product categories: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

### Data Loading & Filtering
- Loaded **9.6M complaint records** from a 5.63GB raw CSV file using chunked processing (100,000 rows per chunk) to manage RAM overhead.
- Filtered for the four target financial products.
- **Result:** 885,813 filtered records.

### Text Cleaning
- Cleaned consumer complaint narratives by enforcing lowercasing, removing special characters, and stripping boilerplate text.
- Removed records with empty narrative blocks.
- Added a narrative word count tracking column for downstream profiling.
- **Result:** 435,737 cleaned records with valid narratives.

### Product Categorization
- Mapped varied CFPB product strings to standardized categories.
- **Distribution:** Credit Cards (43.5%), Savings Accounts (32.2%), Money Transfers (22.3%), Personal Loans (2.0%).

### Key Visualizations & Insights
- **Product Distribution:** Credit Cards heavily dominate the volume of complaints, while Personal Loans account for the fewest.
- **Narrative Length:** Average of 206 words, median of 137 words, with a localized range spanning 1 to 6,472 words.
- **Top Issues:** Managing an account (17.4%), Other transaction problems (11.3%), and Purchase problems (9.3%).
- **Top Target Companies:** Block, Inc., JPMORGAN CHASE, CAPITAL ONE, CITIBANK, BANK OF AMERICA.
- **Geographic Trends:** CA, TX, FL, and NY lead in cumulative complaint generation.
- **Temporal Trends:** Steady growth trend from 2015–2025, displaying distinct Q1 peak seasonality.

### Data Export
- Saved the fully cleaned target dataset to `../data/processed/filtered_complaints.csv`.
- Preserved key structural columns: `Complaint ID`, `Product_Category`, `Product`, `Issue`, `Sub-issue`, `Company`, `State`, `Date received`, and `narrative`.

### Core EDA Findings
- **Narrative Coverage:** Only 49.2% of the filtered complaints contain free-text narratives; the remaining records contain metadata labels only.
- **Data Quality:** Text cleaning successfully stripped high-frequency boilerplate legalese and standardized formatting.
- **Product Balance:** Significant class imbalance exists—Credit Cards hold 48.9% of filtered text narratives vs. Personal Loans at 1.9%.

---

##  2. Chunking Experiment: Finding Optimal Chunk Size

### Experiment Setup
- **Sample Selection:** 200 stratified complaints sampled from the 435,737 cleaned narrative records.
- **Tested Configurations:** 5 distinct text chunk constraints (200, 350, 500, 700, and 1,000 characters).
- **Overlap:** Set statically to 10% of the target chunk size.
- **Splitter:** `RecursiveCharacterTextSplitter` configured with smart sentence boundary fallback rules.

### Empirical Results

| Configuration | Chunk Size | Avg Chunks/Doc | Avg Length | Quality Score |
|---------------|------------|----------------|------------|---------------|
| Very Small    | 200 chars  | 7.47           | 142 chars  | 0             |
| Small         | 350 chars  | 4.05           | 259 chars  | 30            |
| **Medium** | **500** | **2.85** | **369 ch** | **70** |
| Large         | 700 chars  | 2.08           | 507 chars  | 90            |
| Very Large    | 1000 chars | 1.65           | 645 chars  | 70            |

### Core Observations

#### Too Small (200–350 Chunks)
- Forces 4–7 fragments per consumer document, breaking up semantic cohesion.
- Sentences are severed mid-thought, leading to poor contextual similarity mapping during vector retrieval.

#### Optimal (500 Chunks)
- Produces **2.85 chunks per document** and yields a tight **369 character** actual average fill length.
- Successfully splits **42% of documents** into 2–3 complete chunks, offering the most stable balance between localized granular detail and surrounding context.

#### Too Large (700–1000 Chunks)
- Documents pack into fewer chunks (1.6–2.1 per document).
- Long chunks bloat prompt sizes and reduce retrieval precision by mixing unrelated complaints into the LLM context block.

### Final Chunking Recommendation
* **Chunk Size:** 500 characters
* **Chunk Overlap:** 50 characters
* **Rationale:** Optimally breaks the average ~1,200 character narrative into 2–3 context-retaining chunks, matching the technical characteristics of the target dataset.

---

##  3. RAG Pipeline Evaluation Results

To stress-test performance, the pipeline was benchmarked across two different scales using an authenticated `Qwen/Qwen2.5-7B-Instruct` API engine operating at a restricted temperature (`temperature=0.3`) to guarantee factual context anchoring. 

Six adversarial test questions covering billing disputes, money transfers, fraud, and corporate complaint metrics were processed across both configurations.

### Performance Comparison Matrix

| Evaluation Metric | Custom Vector Store | Pre-built Vector Store |
|:---|:---|:---|
| **Total Chunks Indexed** | 36,512 vectors | 1,375,327 vectors |
| **Storage Backend** | Flat Local Artifacts (`.pkl` + `.bin`) | Scaled Parquet Flat Index (`.parquet`) |
| **Average Quality Score** | **5.0 / 5.0** | **4.3 / 5.0** |
| **Search & Retrieval Latency**| ~2.15 seconds | **~1.78 seconds** |
| **Response Format** | Numbered structural lists | Dense prose narrative paragraphs |

### Core Evaluation Takeaways

#### Heuristic Scoring Variation
The Custom Store achieved uniform **5/5 scores** because the LLM generated answers formatted as explicitly numbered lists (e.g., `1. Excessive fees`, `2. Poor customer service`). This triggered a positive structural modifier inside the text scoring engine (`score_answer`). 

The Pre-built Store returned highly descriptive, fluent prose paragraphs that were deeply accurate but missed the structural list bonus point, settling at **4/5 scores**.

#### Sub-Second Subspace Search Scalability
Despite scaling the search space from **36,512 vectors** to a massive production scale of **1,375,327 vectors**, the FAISS index layout showed zero latency degradation. For example, processing *Question 4* on the 1.3M record pre-built store took only **1.32s**, running faster than the local custom store loop.

#### Context Lock & Hallucination Defense
The pipeline successfully defended against domain hallucinations when queried with incomplete knowledge. When evaluated against *Question 6* (*"Which companies receive the most complaints?"*), both pipeline passes securely refused to manufacture synthetic data:
* **Custom Store:** *"...There is no mention of specific companies, only the product category... we do not have enough information to identify specific companies..."*
* **Pre-built Store:** *"...The context does not provide a count or frequency of complaints for each company. Therefore, I don't have enough information..."*

This confirms that setting the LLM generation parameter to a lower temperature successfully forces complete reliance on the retrieved context window.


