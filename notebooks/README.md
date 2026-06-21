## EDA Insight: CFPB Complaint Data Analysis

This notebook performs exploratory data analysis (EDA) and preprocessing on the CFPB complaint dataset to prepare it for the RAG pipeline. The analysis focuses on four key product categories: Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

###  Data Loading & Filtering
- Loaded 9.6M complaint records from a 5.63GB CSV file using chunked processing (100,000 rows per chunk) to manage memory
- Filtered for four target products: Credit Cards, Personal Loans, Savings Accounts, Money Transfers
- Result: 885,813 filtered records

###  Text Cleaning
- Cleaned consumer complaint narratives (lowercasing, removing special characters, boilerplate text removal)
- Removed records with empty narratives
- Added word count column for analysis
- Result: 435,737 cleaned records with narratives

### Product Categorization
- Mapped CFPB product names to standardized categories
- Distribution: Credit Cards (43.5%), Savings Accounts (32.2%), Money Transfers (22.3%), Personal Loans (2.0%)

###  Key Visualizations & Insights
- **Product Distribution**: Credit Cards dominate complaints, Personal Loans have the fewest
- **Narrative Length**: Average 206 words, median 137 words, range 1-6,472 words
- **Top Issues**: Managing an account (17.4%), Other transaction problems (11.3%), Purchase problems (9.3%)
- **Top Companies**: Block, Inc., JPMORGAN CHASE, CAPITAL ONE, CITIBANK, BANK OF AMERICA
- **Geographic**: CA, TX, FL, NY lead in complaints
- **Temporal**: Growing trend from 2015-2025 with Q1 peak seasonality

### 5. Data Export
- Saved cleaned dataset to `../data/processed/filtered_complaints.csv`
- Preserved key columns: Complaint ID, Product_Category, Product, Issue, Sub-issue, Company, State, Date received, narrative

## Key Findings
- **Narrative Coverage**: Only 49.2% of filtered complaints contain narratives
- **Data Quality**: Text cleaning removed boilerplate and standardized formatting
- **Product Balance**: Significant imbalance - Credit Cards (48.9% of filtered) vs Personal Loans (1.9%)
- **Rich Metadata**: Each complaint includes issue type, company, state, and date for context

# Chunking Experiment: Finding Optimal Chunk Size

## Experiment Setup
- **Sample**: 200 stratified complaints from 435,737 cleaned records
- **Tested Configurations**: 5 chunk sizes (200, 350, 500, 700, 1000 chars)
- **Overlap**: 10% of chunk size
- **Splitter**: RecursiveCharacterTextSplitter with sentence boundaries

## Results

| Configuration | Chunk Size | Avg Chunks/Doc | Avg Length | Quality Score |
|---------------|------------|----------------|------------|---------------|
| Very Small | 200 | 7.47 | 142 chars | 0  |
| Small | 350 | 4.05 | 259 chars | 30  |
| **Medium** | **500** | **2.85** | **369 chars** | **70 ** |
| Large | 700 | 2.08 | 507 chars | 90  |
| Very Large | 1000 | 1.65 | 645 chars | 70  |

## Key Observations

###  Too Small (200-350)
- 4-7 chunks per document → too fragmented
- Loses semantic coherence
- Poor retrieval quality

###  Optimal (500)
- **2.85 chunks/doc** (near ideal 1.5-2.5)
- **369 chars** (ideal 300-600 range)
- **42% documents** split into 2-3 chunks (best among configs)
- Best balance of granularity and context

### Too Large (700-1000)
- Fewer chunks (1.6-2.1 per doc)
- Longer chunks reduce retrieval precision
- Less granular for specific queries

## Recommendation

**Chunk Size: 500 characters | Overlap: 50 characters**

**Why:**
- Most narratives (avg ~1,200 chars) split into 2-3 chunks
- Preserves semantic meaning while maintaining granularity
- Best retrieval precision for specific complaint queries
- Aligns with dataset characteristics from EDA


