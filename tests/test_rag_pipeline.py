"""
Simple unit tests for RAG pipeline.
Tests only the logic, not the actual API calls.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRAGPipeline(unittest.TestCase):
    """Simple tests for RAG pipeline logic."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.mock_faiss = MagicMock()
        self.mock_sentence_transformer = MagicMock()

    def test_format_context_empty(self):
        """Test format_context with empty results."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.chunks = []
        rag.metadatas = []
        rag.index = None
        rag.model = MagicMock()
        rag.k = 5

        result = rag.format_context([])
        self.assertEqual(result, "No relevant information found.")

    def test_format_context_with_results(self):
        """Test format_context with sample results."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.chunks = ["This is a test chunk about credit card fees."]
        rag.metadatas = [{"product_category": "Credit Cards", "issue": "Fees"}]

        results = [
            {
                "chunk": "This is a test chunk about credit card fees.",
                "metadata": {"product_category": "Credit Cards", "issue": "Fees"},
                "distance": 0.5,
            }
        ]

        result = rag.format_context(results)

        self.assertIn("Credit Cards", result)
        self.assertIn("Fees", result)
        self.assertIn("This is a test chunk", result)

    def test_answer_structure(self):
        """Test that answer returns expected structure."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.use_llm = False
        rag.model = MagicMock()

        def mock_retrieve(question):
            return [
                {
                    "chunk": "Test chunk",
                    "metadata": {"product_category": "Credit Cards", "issue": "Fees"},
                    "distance": 0.5,
                }
            ]

        rag.retrieve = mock_retrieve
        rag.format_context = lambda x: "Formatted context"
        rag.generate_answer = lambda q, c: "Generated answer"

        result = rag.answer("Test question")

        self.assertIn("question", result)
        self.assertIn("answer", result)
        self.assertIn("context", result)
        self.assertIn("sources", result)
        self.assertIn("num_sources", result)
        self.assertEqual(result["question"], "Test question")
        self.assertEqual(result["answer"], "Generated answer")

    def test_search_only(self):
        """Test search_only method returns results."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.k = 5

        def mock_retrieve(question):
            return [{"chunk": f"Result {i}", "distance": 0.5} for i in range(rag.k)]

        rag.retrieve = mock_retrieve

        results = rag.search_only("Test question", k=3)
        self.assertEqual(len(results), 3)

    def test_load_custom_vector_store_raises_error(self):
        """Test that loading non-existent vector store raises error."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)

        with self.assertRaises(FileNotFoundError):
            rag.load_custom_vector_store("non_existent_path")

    def test_load_prebuilt_vector_store_raises_error(self):
        """Test that loading non-existent pre-built store raises error."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)

        with self.assertRaises(FileNotFoundError):
            rag.load_prebuilt_vector_store("non_existent_file.parquet")

    def test_retrieve_handles_empty_index(self):
        """Test retrieve handles case when no chunks exist."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.model = MagicMock()
        rag.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        rag.index = MagicMock()

        # Return index -1 to simulate no results found
        # But also return a valid distance so the loop runs
        rag.index.search.return_value = (
            np.array([[0.5]]),
            np.array([[0]]),  # Return index 0 instead of -1
        )
        rag.chunks = []
        rag.metadatas = []
        rag.k = 5

        results = rag.retrieve("Test question")
        # Since idx=0 and len(chunks)=0, the condition idx < len(self.chunks) fails
        # So results should be empty
        self.assertEqual(results, [])

    def test_generate_answer_without_llm(self):
        """Test generate_answer when LLM is disabled."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.use_llm = False

        result = rag.generate_answer("Test", "Context")
        self.assertEqual(result, "Based on the retrieved context:\n\nContext")

    def test_retrieve_with_valid_index(self):
        """Test retrieve with valid index returns correct chunk."""
        from src.rag_pipeline import RAGPipeline

        rag = RAGPipeline.__new__(RAGPipeline)
        rag.model = MagicMock()
        rag.model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        rag.index = MagicMock()

        # Return index 0 with valid distance
        rag.index.search.return_value = (np.array([[0.5]]), np.array([[0]]))
        rag.chunks = ["This is a test chunk."]
        rag.metadatas = [{"product_category": "Credit Cards", "issue": "Fees"}]
        rag.k = 5

        results = rag.retrieve("Test question")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk"], "This is a test chunk.")
        self.assertEqual(results[0]["metadata"]["product_category"], "Credit Cards")


if __name__ == "__main__":
    unittest.main()
