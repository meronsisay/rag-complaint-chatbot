"""
RAG Pipeline for complaint analysis.
Handles retrieval, prompt engineering, and LLM generation.
"""

import os
import pickle
from typing import Dict, List, Optional

import faiss
import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ─── ADDED: Native support for loading a local .env file ───
try:
    from dotenv import load_dotenv

    load_dotenv()  # This scans for a .env file and injects its values into os.environ
except ImportError:
    print("'python-dotenv' package not found. Reading from system environment variables only.")


class RAGPipeline:

    def __init__(
        self,
        vector_store_path="vector_store",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        k=5,
        use_prebuilt=False,
        use_llm=False,
        use_api=False,
        api_model="Qwen/Qwen2.5-7B-Instruct",
        api_token=None,
    ):
        """Initialize the RAG pipeline."""
        print("=" * 60)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 60)

        self.k = k
        self.use_llm = use_llm

        # ─── UPDATED: Clean token fallback architecture prioritizing .env ───
        self.use_api = use_api
        self.api_model = api_model
        self.api_token = (
            api_token
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGINGFACE_CO_API_TOKEN")
        )

        # Load vector store
        if use_prebuilt:
            self.load_prebuilt_vector_store(vector_store_path)
        else:
            self.load_custom_vector_store(vector_store_path)

        # Load embedding model
        self.load_embedding_model(model_name)

        # Load LLM (if requested)
        if use_llm:
            if use_api:
                self.load_llm_api(model=self.api_model, token=self.api_token)
            else:
                self.load_llm()

        print("-" * 60)

    # =============================================================
    # VECTOR STORE LOADING
    # =============================================================

    def load_prebuilt_vector_store(self, parquet_path):
        """Load pre-built embeddings from parquet file."""
        print(f"\n Loading pre-built vector store from: {parquet_path}")

        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Vector store not found: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        print(f"   Loaded {len(df):,} records")
        print(f"   Columns: {df.columns.tolist()}")
        print(
            f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        print("\n   Extracting chunks and metadata...")
        self.chunks = df["document"].tolist()

        metadata_cols = [
            col for col in df.columns if col not in ["document", "embedding"]
        ]
        self.metadatas = df[metadata_cols].to_dict("records")

        print(f"   Chunks: {len(self.chunks):,}")
        print(f"   Metadata columns: {metadata_cols}")

        print("\n   Creating FAISS index...")
        embeddings = np.vstack(df["embedding"].values)
        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype("float32"))

        print(f"   Embedding dimension: {self.dimension}")
        print(f"   Index size: {self.index.ntotal:,} vectors")
        print(f"   Index memory: {embeddings.nbytes / 1024**2:.2f} MB")

        self.embeddings = embeddings
        print(f"   Vector store loaded successfully!")

    def load_custom_vector_store(self, folder_path):
        """Load custom FAISS index and artifacts (from Task 2)."""
        print(f"\n Loading custom vector store from: {folder_path}")

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Vector store not found: {folder_path}")

        index_path = os.path.join(folder_path, "faiss_index.bin")
        self.index = faiss.read_index(index_path)
        print(f"   Loaded FAISS index: {self.index.ntotal:,} vectors")

        with open(os.path.join(folder_path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"   Loaded {len(self.chunks):,} chunks")

        with open(os.path.join(folder_path, "metadatas.pkl"), "rb") as f:
            self.metadatas = pickle.load(f)
        print(f"   Loaded {len(self.metadatas):,} metadata records")

        self.dimension = self.index.d
        print(f"   Vector store loaded successfully!")

    # =========================================================================
    # EMBEDDING MODEL
    # =========================================================================

    def load_embedding_model(self, model_name):
        """Load the embedding model."""
        print(f"\n Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"   Model loaded")
        print(f"   Embedding dimension: {self.model.get_embedding_dimension()}")

    # =======================================================
    # LLM LOADING WITH API SUPPORT
    # =======================================================

    def load_llm(self):
        """Load LLM locally (requires GPU)."""
        try:
            print(f"\n Loading local LLM...")
            self.llm = pipeline(
                "text-generation",
                model="mistralai/Mistral-7B-Instruct-v0.2",
                device_map="auto",
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
            print(f"   LLM loaded: mistralai/Mistral-7B-Instruct-v0.2")
            self.llm_type = "local"
        except Exception as e:
            print(f"   Could not load local LLM: {e}")
            print("   Falling back to context-only mode")
            self.use_llm = False
            self.llm = None
            self.llm_type = None

    def load_llm_api(self, model=None, token=None):
        """Load LLM via Hugging Face Inference API."""
        model = model or self.api_model
        token = token or self.api_token

        try:
            print(f"\n Loading LLM via API: {model}")

            if token:
                self.llm_client = InferenceClient(model=model, token=token)
                print("   Using authenticated access from environment configuration")
            else:
                self.llm_client = InferenceClient(model=model)
                print("   Using public access (may have rate limits)")

            self.use_llm = True
            self.llm_type = "api"
            self.llm = None
            print(f"   API client ready: {model}")
            return True

        except Exception as e:
            print(f"   Could not load API LLM: {e}")
            self.use_llm = False
            self.llm_client = None
            self.llm_type = None
            return False

    # =============================
    # RETRIEVAL
    # =============================

    def retrieve(self, question: str) -> List[Dict]:
        """Retrieve top-k relevant chunks for a user question."""
        question_embedding = self.model.encode(
            [question], convert_to_numpy=True
        )

        distances, indices = self.index.search(
            question_embedding.astype("float32"), self.k
        )

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "metadata": self.metadatas[idx],
                        "distance": float(distance),
                    }
                )

        return results

    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not results:
            return "No relevant information found."

        context_parts = []
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})

            if isinstance(meta, dict) and "metadata" in meta:
                inner_meta = meta["metadata"]
                if isinstance(inner_meta, dict):
                    meta = inner_meta

            product = meta.get(
                "product_category", meta.get("product", "Unknown Product")
            )
            issue = meta.get("issue", "Unknown Issue")

            context_parts.append(
                f"[Excerpt {i}] "
                f"Product: {product} | "
                f"Issue: {issue}\n"
                f"{result['chunk']}\n"
            )

        return "\n".join(context_parts)

    # ==================================================
    # GENERATION WITH CHAT COMPLETIONS API
    # ==================================================

    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        if not self.use_llm:
            return f"Based on the retrieved context:\n\n{context}"

        if (
            self.llm_type == "api"
            and hasattr(self, "llm_client")
            and self.llm_client
        ):
            try:
                system_instruction = (
                    "You are a financial analyst assistant for CrediTrust Financial. "
                    "Your task is to answer questions about customer complaints. "
                    "Use ONLY the following context to formulate your answer. "
                    "If the context doesn't contain the answer, state that you don't have enough information."
                )
                user_content = f"Context:\n{context}\n\nQuestion: {question}"

                completion = self.llm_client.chat.completions.create(
                    model=self.api_model,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content},
                    ],
                    max_tokens=300,
                    temperature=0.3,
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                return f"Error generating answer via API: {e}\n\nContext:\n{context}"

        if self.llm_type == "local" and self.llm is not None:
            prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            try:
                response = self.llm(prompt, max_new_tokens=300)[0][
                    "generated_text"
                ]
                if "Answer:" in response:
                    return response.split("Answer:")[-1].strip()
                return response.strip()
            except Exception as e:
                return f"Error generating answer: {e}\n\nContext:\n{context}"

        return f"Context:\n{context}"

    # ============================
    # MAIN METHODS
    # ============================

    def answer(self, question: str) -> Dict:
        """Full RAG pipeline: Retrieve + Format + Generate."""
        results = self.retrieve(question)
        context = self.format_context(results)
        answer = self.generate_answer(question, context)

        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": results,
            "num_sources": len(results),
        }

    def search_only(self, question: str, k: int = None) -> List[Dict]:
        """Retrieve only (no generation). Useful for debugging."""
        if k:
            original_k = self.k
            self.k = k
            results = self.retrieve(question)
            self.k = original_k
            return results
        return self.retrieve(question)


# =========================================================================
# TEST EXECUTOR
# =========================================================================

if __name__ == "__main__":

    # To run this successfully, ensure you have a '.env' file in the root folder with:
    # HF_TOKEN=hf_yourActualTokenString Here

    rag = RAGPipeline(
        vector_store_path="vector_store",
        use_prebuilt=False,
        use_llm=True,
        use_api=True,
        api_model="Qwen/Qwen2.5-7B-Instruct",
    )

    question = "Why are customers unhappy with credit card fees?"
    result = rag.answer(question)

    print("\n" + "=" * 60)
    print(f"QUESTION: {question}")
    print("=" * 60)
    print(f"\n ANSWER:")
    print(result["answer"])