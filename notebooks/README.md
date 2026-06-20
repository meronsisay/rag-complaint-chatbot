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

## Next Steps
The cleaned dataset (435,737 records) is ready for **Task 2**: Text Chunking, Embedding, and Vector Store Indexing for the RAG pipeline.

## Files Generated
- `../data/processed/filtered_complaints.csv`: Cleaned dataset with 435,737 records
