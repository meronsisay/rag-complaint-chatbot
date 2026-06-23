"""
Text chunking, embedding, and vector store indexing module.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.model_selection import train_test_split
import faiss
import pickle
import os
from tqdm import tqdm


class ComplaintChunker:
    def __init__(
        self,
        data_path="data/processed/filtered_complaints.csv",
        chunk_size=500,
        chunk_overlap=50,
    ):
        print("=" * 60)
        print("INITIALIZING CHUNKER")
        print("=" * 60)

        self.df = pd.read_csv(data_path)
        print(f"Loaded {len(self.df):,} records")
        print(f"Columns: {self.df.columns.tolist()}")

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print(f" Loaded embedding model: all-MiniLM-L6-v2")
        print(
            f"   Embedding dimension: {self.model.get_sentence_embedding_dimension()}"
        )

        # Parameters are now dynamic and configurable
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        print(
            f" Configured text splitter: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}"
        )
        print("-" * 60)

    def sample(self, size=12000, random_state=42):
        """Create stratified sample using train_test_split."""
        print("\n" + "=" * 60)
        print("STRATIFIED SAMPLING")
        print("=" * 60)
        print(f"Target sample size: {size:,} records")
        print(f"Random seed: {random_state}")

        # Original distribution
        category_counts = self.df["Product_Category"].value_counts()
        print("\nOriginal Distribution:")
        for cat, count in category_counts.items():
            pct = count / len(self.df) * 100
            print(f"  {cat}: {count:,} ({pct:.1f}%)")

        # Calculate sample size per category proportionally
        sample_sizes = {}
        for category in self.df["Product_Category"].unique():
            category_count = len(self.df[self.df["Product_Category"] == category])
            # Proportional sample size, capped at category size
            sample_size = int(size * category_count / len(self.df))
            sample_size = min(sample_size, category_count)
            sample_sizes[category] = sample_size

        # Perform stratified sampling
        sampled_dfs = []
        for category, n_samples in sample_sizes.items():
            category_df = self.df[self.df["Product_Category"] == category]
            if n_samples > 0:
                sampled, _ = train_test_split(
                    category_df,
                    train_size=n_samples,
                    random_state=random_state,
                    stratify=None,  # Already filtered by category
                )
                sampled_dfs.append(sampled)

        self.sample_df = pd.concat(sampled_dfs, ignore_index=True)

        # If we didn't get enough samples, add random extra
        if len(self.sample_df) < size:
            remaining = self.df[~self.df.index.isin(self.sample_df.index)]
            needed = size - len(self.sample_df)
            extra = remaining.sample(
                min(needed, len(remaining)), random_state=random_state
            )
            self.sample_df = pd.concat([self.sample_df, extra], ignore_index=True)

        print(f"\n Sampled {len(self.sample_df):,} records")
        print("\nSampled Distribution:")
        for cat, count in self.sample_df["Product_Category"].value_counts().items():
            pct = count / len(self.sample_df) * 100
            print(f"  {cat}: {count:,} ({pct:.1f}%)")

        return self.sample_df

    def chunk(self):
        """Chunk narratives."""
        print("\n" + "=" * 60)
        print("TEXT CHUNKING")
        print("=" * 60)
        print(f"Chunk size: {self.chunk_size} characters")
        print(f"Chunk overlap: {self.chunk_overlap} characters")

        self.chunks = []
        self.metadatas = []
        total_chunks_created = 0

        for idx, row in tqdm(
            self.sample_df.iterrows(), total=len(self.sample_df), desc="Chunking"
        ):
            text = str(row["narrative"])
            if text.strip():
                chunks = self.splitter.split_text(text)
                for i, chunk in enumerate(chunks):
                    self.chunks.append(chunk)
                    self.metadatas.append(
                        {
                            "complaint_id": row["Complaint ID"],
                            "product_category": row["Product_Category"],
                            "issue": row.get("Issue", ""),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        }
                    )
                total_chunks_created += len(chunks)

        print(f"\n Created {len(self.chunks):,} chunks")
        print(
            f"   Average chunks per complaint: {len(self.chunks)/len(self.sample_df):.1f}"
        )
        print(f"   Total chunks created: {total_chunks_created:,}")

        # Show sample chunks
        print("\nSample Chunks:")
        for i in range(min(3, len(self.chunks))):
            print(f"\n   Chunk {i+1} (ID: {self.metadatas[i]['complaint_id']}):")
            print(f"    {self.chunks[i][:150]}...")

        return self.chunks, self.metadatas

    def embed(self, batch_size=32):
        """Generate embeddings."""
        print("\n" + "=" * 60)
        print("GENERATING EMBEDDINGS")
        print("=" * 60)
        print(f"Total chunks to embed: {len(self.chunks):,}")
        print(f"Batch size: {batch_size}")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        embeddings = []
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding"):
            batch = self.chunks[i : i + batch_size]
            embeds = self.model.encode(
                batch, show_progress_bar=False, convert_to_numpy=True
            )
            embeddings.append(embeds)

        self.embeddings = np.vstack(embeddings)

        print(f"\n Generated embeddings")
        print(f"   Shape: {self.embeddings.shape}")
        print(f"   Memory usage: {self.embeddings.nbytes / 1024**2:.2f} MB")
        print(f"   Data type: {self.embeddings.dtype}")

        return self.embeddings

    def create_index(self, save_path="../vector_store"):
        """Create and save FAISS index."""
        print("\n" + "=" * 60)
        print("CREATING FAISS INDEX")
        print("=" * 60)
        print(f"Save path: {save_path}")
        print(f"Vectors to index: {self.embeddings.shape[0]:,}")
        print(f"Vector dimension: {self.embeddings.shape[1]}")

        os.makedirs(save_path, exist_ok=True)

        # Create index
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        print(f"Created FAISS index (L2 distance)")

        # Add vectors
        print("Adding vectors to index...")
        index.add(self.embeddings.astype("float32"))
        print(f"Added {index.ntotal:,} vectors")

        # Save files
        print("\nSaving files...")

        # Save index
        index_path = f"{save_path}/faiss_index.bin"
        faiss.write_index(index, index_path)
        print(f"   Saved index: {index_path}")

        # Save chunks
        chunks_path = f"{save_path}/chunks.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"  Saved chunks: {chunks_path}")

        # Save metadata
        metas_path = f"{save_path}/metadatas.pkl"
        with open(metas_path, "wb") as f:
            pickle.dump(self.metadatas, f)
        print(f"   Saved metadata: {metas_path}")

        # Show file sizes
        print("\nFile Sizes:")
        for file in [index_path, chunks_path, metas_path]:
            size_mb = os.path.getsize(file) / 1024**2
            print(f"  {os.path.basename(file)}: {size_mb:.2f} MB")

        self.index = index
        print(f"\n Vector store saved to {save_path}")
        return index

    def test_search(
        self, query="Why are customers unhappy with credit card fees?", k=5
    ):
        """Test retrieval with a sample query."""
        print("\n" + "=" * 60)
        print("TESTING RETRIEVAL")
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Results: {k}")
        print("-" * 60)

        # Embed query
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Search
        distances, indices = self.index.search(query_embedding.astype("float32"), k)

        # Display results
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            print(f"\nResult {i+1} (Distance: {dist:.4f})")
            print(f"  ID: {self.metadatas[idx]['complaint_id']}")
            print(f"  Product: {self.metadatas[idx]['product_category']}")
            print(f"  Issue: {self.metadatas[idx]['issue']}")
            print(f"  Chunk: {self.chunks[idx][:200]}...")

        return indices, distances

    def run(self, size=12000, test_query=None):
        """Run full pipeline with optional test query."""
        print("\n" + "=" * 60)
        print("RUNNING FULL PIPELINE")
        print("=" * 60)

        self.sample(size)
        self.chunk()
        self.embed()
        self.create_index(save_path="vector_store")

        # Test retrieval if query provided
        if test_query:
            self.test_search(test_query)
        else:
            # Default test queries
            test_queries = [
                "Why are customers unhappy with credit card fees?",
                "What are the main issues with money transfers?",
                "How do customers complain about unauthorized charges?",
            ]
            print("\n" + "=" * 60)
            print("RUNNING TEST QUERIES")
            print("=" * 60)
            for query in test_queries:
                self.test_search(query, k=3)

        print(f"Sample size: {len(self.sample_df):,}")
        print(f"Chunks created: {len(self.chunks):,}")
        print(f"Embedding dimension: {self.embeddings.shape[1]}")
        print(f"FAISS index size: {self.index.ntotal:,}")
        print(f"Files saved in: ../vector_store_sz_{self.chunk_size}/")
        print("=" * 60)

        return self.index


# Execution fallback when running script file directly
if __name__ == "__main__":
    chunker = ComplaintChunker(chunk_size=500, chunk_overlap=50)
    chunker.run(size=12000)
