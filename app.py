"""
Gradio Chat Interface for RAG Complaint Analysis System.
Allows users to ask questions about customer complaints and get AI-generated answers.
"""

import os
import sys
import time
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG pipeline
from src.rag_pipeline import RAGPipeline

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_PREBUILT = True
VECTOR_STORE_PATH = "vector_store" if not USE_PREBUILT else "data/processed/complaint_embeddings.parquet"

# Initialize RAG pipeline
print("=" * 60)
print("LOADING RAG PIPELINE FOR UI")
print("=" * 60)

rag = RAGPipeline(
    vector_store_path=VECTOR_STORE_PATH,
    use_prebuilt=USE_PREBUILT,
    use_llm=True,
    use_api=True,
    api_model="Qwen/Qwen2.5-7B-Instruct",
)

print(f" RAG Pipeline loaded successfully!")
print(f"   Vector store: {len(rag.chunks):,} chunks")
print("=" * 60)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def format_sources(sources):
    """Format sources cleanly based on whether custom or pre-built mode is active."""
    if not sources:
        return "No sources retrieved."
    
    formatted = ""
    for i, source in enumerate(sources, 1):
        chunk = source['chunk']
        distance = source.get('distance', 0)
        
        formatted += f"### Source {i}\n"
        formatted += f"**Relevance Confidence Metric:** `{1 - distance:.3f}`\n\n"
        
        # DYNAMIC ADJUSTMENT: Only extract metadata fields if we are NOT using the prebuilt dataset
        if not USE_PREBUILT:
            product = source.get('metadata', {}).get('product_category', 'Unknown')
            issue = source.get('metadata', {}).get('issue', 'Unknown')
            formatted += f"**Product Category:** `{product}`\n\n"
            formatted += f"**Primary Issue:** `{issue}`\n\n"
            
        formatted += f"**Document Excerpt Context:**\n> *{chunk.strip()}*\n\n"
        
        if i < len(sources):
            formatted += "---\n\n"
    
    return formatted


def answer_question(question, history):
    """
    NON-STREAMING: Process user question and return answer with sources all at once.
    """
    if not question or not question.strip():
        return " *Please enter a valid question.*", "", history
    
    result = rag.answer(question)
    answer = result['answer']
    sources = result['sources']
    
    sources_display = format_sources(sources)
    history = history or []
    history.append((question, answer))
    
    return answer, sources_display, history


def answer_with_streaming(question, history):
    """
    STREAMING: Process question and stream answer character by character.
    """
    if not question or not question.strip():
        yield " *Please enter a valid question.*", "", history
        return
    
    history = history or []
    yield " *Processing request... Generating response syntax.*", " *Running semantic search over reference nodes...*", history
    
    result = rag.answer(question)
    answer = result['answer']
    sources = result['sources']
    sources_display = format_sources(sources)
    
    streamed_answer = ""
    for char in answer:
        streamed_answer += char
        yield streamed_answer, sources_display, history
    
    history.append((question, answer))
    yield answer, sources_display, history


def clear_conversation():
    """Clear the chat history safely."""
    return "", "*Ask a question to get started...*", "*Sources will appear here after you ask a question...*", None


# ============================================================================
# SUGGESTED QUESTIONS
# ============================================================================

SUGGESTED_QUESTIONS = [
    "Why are customers unhappy with credit card fees?",
    "What are the main issues with money transfers?",
    "How do customers complain about unauthorized charges?",
    "What are the top complaint issues overall?",
    "How do customers describe fraud or scam experiences?",
    "Which companies receive the most complaints?",
]


# ============================================================================
# NATIVE THEME STYLING OVERRIDES (Fixes contrast and crowding)
# ============================================================================

# Creating a theme palette to balance dark/light text ratios natively
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    block_background_fill="*neutral_50", 
    block_border_width="1px",
    block_label_text_size="*text_sm",
    body_background_fill="*neutral_100",
    layout_gap="14px" 
)

custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
.app-header {
    text-align: center;
    margin-bottom: 10px;
}
.response-container {
    padding: 8px !important;
}
.suggested-btn-group button {
    justify-content: flex-start !important;
    text-align: left !important;
    border-left: 3px solid var(--blue-500) !important;
    margin-bottom: 4px !important;
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="CrediTrust RAG Engine") as demo:
    
    # App Header Section
    with gr.Row(elem_classes="app-header"):
        with gr.Column():
            gr.Markdown(
                """
                #  CrediTrust Audit & Complaint Analysis System
                *Enterprise retrieval framework extracting unstructured insights from Consumer Financial Protection Bureau data repositories.*
                """
            )
            
    # Main Split Screen Interface
    with gr.Row():
        
        # Left Panel: Query Utilities and Parameters
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("###  Inquiry Parameters")
                question_input = gr.Textbox(
                    label="Active Search Vector Query",
                    placeholder="Type an analytical question regarding customer friction...",
                    lines=3,
                    max_lines=6
                )
                
                with gr.Row():
                    submit_btn = gr.Button(" Execute Query", variant="primary")
                    clear_btn = gr.Button(" Reset Context", variant="secondary")
            
            with gr.Group():
                gr.Markdown("###  Engine Options")
                streaming_toggle = gr.Checkbox(
                    label="Enable Token Streaming Output",
                    value=True
                )
            
            # Grouped Shortcut List
            gr.Markdown("###  Recommended Analytical Vectors")
            with gr.Column(elem_classes="suggested-btn-group"):
                suggestion_btns = []
                for q in SUGGESTED_QUESTIONS:
                    btn = gr.Button(q, size="sm", variant="secondary")
                    suggestion_btns.append(btn)
                    
        # Right Panel: Output Generation Window
        with gr.Column(scale=3, elem_classes="response-container"):
            with gr.Tab(" Synthesized Audit Summary"):
                answer_output = gr.Markdown(
                    value="*Awaiting user execution input matrix...*",
                    line_breaks=True
                )
                
            # Embed Sources into a clean Accordion below the response to prevent crowding
            with gr.Accordion(" Extracted Ground-Truth Reference Clusters", open=False):
                sources_output = gr.Markdown(
                    value="*Semantic nodes will be rendered dynamically here following execution validation.*",
                    line_breaks=True
                )

    # Chat history state variable
    chat_history = gr.State([])
    
    # ========================================================================
    # WIRE EVENT HANDLERS
    # ========================================================================
    
    # Submit Click Execution
    submit_btn.click(
        fn=answer_with_streaming,
        inputs=[question_input, chat_history],
        outputs=[answer_output, sources_output, chat_history]
    )
    
    # Textbox Enter Key Submission Execution
    question_input.submit(
        fn=answer_with_streaming,
        inputs=[question_input, chat_history],
        outputs=[answer_output, sources_output, chat_history]
    )
    
    # System Context Purge Button Event
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[question_input, answer_output, sources_output, chat_history]
    )
    
    # Suggested Analytical Question Trigger Sequences
    for btn, question in zip(suggestion_btns, SUGGESTED_QUESTIONS):
        btn.click(
            fn=lambda q=question: q,
            inputs=[],
            outputs=[question_input]
        ).then(
            fn=answer_with_streaming,
            inputs=[question_input, chat_history],
            outputs=[answer_output, sources_output, chat_history]
        )
    
    # Clean Footer
    gr.Markdown(
        """
        ---
        <p align="center" style="font-size: 0.85em; opacity: 0.8;">
            <b>LLM Node Architecture:</b> Qwen2.5-7B-Instruct API &nbsp;|&nbsp; 
            <b>Vector Store Core:</b> Index FAISS Matrix &nbsp;|&nbsp; 
            <b>Audit Data:</b> CFPB Consumer Ledger
        </p>
        """
    )


# ============================================================================
# LAUNCH INSTANCE
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=False
    )