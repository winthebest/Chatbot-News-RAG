"""
Gradio web interface for RAG News Chatbot.
Provides a user-friendly web UI for interacting with the RAG system.
"""

import gradio as gr
import ollama
from typing import List, Dict, Tuple, Optional

from .retriever import init_rag_components, retrieve_news
from .chatbot import build_prompt, answer_with_ollama


class RAGChatbotUI:
    """Gradio UI wrapper for RAG Chatbot."""
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        api_key: Optional[str] = None,
        use_reranker: bool = True,
        reranker_model: str = "BAAI/bge-reranker-base",
        ollama_model: str = "qwen2.5:7b",
        collection: str = "vnexpress_news",
    ):
        """Initialize RAG components."""
        self.ollama_model = ollama_model
        self.collection = collection
        
        print("Initializing RAG components...")
        self.model, self.reranker, self.client = init_rag_components(
            model_name=model_name,
            device=device,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            api_key=api_key,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
        )
        print("RAG components initialized successfully!")
    
    def format_sources(self, contexts: List[Dict]) -> str:
        """Format source references for display."""
        if not contexts:
            return "Không có nguồn tham khảo."
        
        sources_html = "<div style='margin-top: 20px;'>"
        sources_html += "<h3>📚 Nguồn tham khảo:</h3>"
        
        for i, ctx in enumerate(contexts, start=1):
            title = ctx.get("title") or "(Không có tiêu đề)"
            url = ctx.get("url") or ""
            score = ctx.get("score", 0.0)
            vector_score = ctx.get("vector_score")
            rerank_score = ctx.get("rerank_score")
            lang = ctx.get("lang", "?")
            
            # Format scores
            score_text = ""
            if rerank_score is not None and vector_score is not None:
                score_text = f"Rerank: {rerank_score:.3f} | Vector: {vector_score:.3f}"
            elif vector_score is not None:
                score_text = f"Vector: {vector_score:.3f}"
            else:
                score_text = f"Score: {score:.3f}"
            
            # Color based on score
            if score > 0.7:
                score_color = "#28a745"  # Green
            elif score > 0.5:
                score_color = "#ffc107"  # Yellow
            else:
                score_color = "#dc3545"  # Red
            
            sources_html += f"""
            <div style='margin-bottom: 15px; padding: 10px; border-left: 3px solid {score_color}; background-color: #f8f9fa;'>
                <strong>[{i}] {title}</strong><br>
                <span style='color: #6c757d; font-size: 0.9em;'>
                    {score_text} | Lang: {lang}
                </span><br>
                {f'<a href="{url}" target="_blank" style="color: #007bff;">{url}</a>' if url else ''}
            </div>
            """
        
        sources_html += "</div>"
        return sources_html
    
    def chat(
        self,
        question: str,
        history: List[Dict[str, str]],
        top_k: int,
        use_reranker: bool,
        initial_candidates: int,
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process user question and return answer with sources.
        
        Args:
            question: User question
            history: Chat history (list of dicts with 'role' and 'content')
            top_k: Number of results to retrieve
            use_reranker: Whether to use reranker
            initial_candidates: Number of initial candidates for reranking
        
        Returns:
            Tuple of (sources_html, updated_history)
        """
        if not question.strip():
            return "", history
        
        try:
            # Retrieve context
            contexts = retrieve_news(
                client=self.client,
                model=self.model,
                question=question,
                collection=self.collection,
                top_k=top_k,
                reranker=self.reranker if use_reranker else None,
                initial_candidates=initial_candidates if use_reranker else None,
            )
            
            # Build prompt
            prompt = build_prompt(question, contexts)
            
            # Generate answer
            answer = answer_with_ollama(prompt, model_name=self.ollama_model)
            
            # Format sources
            sources_html = self.format_sources(contexts)
            
            # Update history with messages format
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            
            return sources_html, history
            
        except Exception as e:
            error_msg = f"Lỗi: {str(e)}"
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def create_interface(self) -> gr.Blocks:
        """Create and return Gradio interface."""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        """
        
        with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as interface:
            # Header
            gr.HTML("""
                <div class="main-header">
                    <h1> RAG News Chatbot</h1>
                    <p>Hỏi đáp thông minh về tin tức công nghệ từ VNExpress</p>
                </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="💬 Cuộc trò chuyện",
                        height=500,
                        show_copy_button=True,
                        avatar_images=(None, "🤖"),
                        type="messages",
                    )
                    
                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Nhập câu hỏi của bạn",
                            placeholder="Ví dụ: Google phát triển AI tiếng Việt như thế nào?",
                            lines=2,
                            scale=4,
                        )
                        submit_btn = gr.Button("Gửi", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Xóa lịch sử", variant="secondary")
                
                with gr.Column(scale=1):
                    # Settings panel
                    gr.Markdown("### ⚙️ Cài đặt")
                    
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Số lượng kết quả (Top-K)",
                        info="Số lượng tài liệu tham khảo để trả lời",
                    )
                    
                    use_reranker_checkbox = gr.Checkbox(
                        value=True,
                        label="Sử dụng Re-ranking",
                        info="Cải thiện độ chính xác (chậm hơn)",
                    )
                    
                    initial_candidates_slider = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Số ứng viên ban đầu",
                        info="Số lượng tài liệu lấy trước khi re-rank",
                        visible=True,
                    )
                    
                    # Show/hide initial_candidates based on reranker checkbox
                    use_reranker_checkbox.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=use_reranker_checkbox,
                        outputs=initial_candidates_slider,
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 📊 Thông tin hệ thống")
                    gr.Markdown(f"""
                    - **Embedding Model**: BAAI/bge-m3
                    - **Reranker**: BAAI/bge-reranker-base
                    - **LLM**: {self.ollama_model}
                    - **Collection**: {self.collection}
                    """)
            
            # Sources display
            sources_output = gr.HTML(
                label="📚 Nguồn tham khảo",
                visible=True,
            )
            
            # Event handlers
            def respond(question, history, top_k, use_reranker, initial_candidates):
                sources, updated_history = self.chat(
                    question, history, top_k, use_reranker, initial_candidates
                )
                return updated_history, sources, ""
            
            submit_btn.click(
                fn=respond,
                inputs=[question_input, chatbot, top_k_slider, use_reranker_checkbox, initial_candidates_slider],
                outputs=[chatbot, sources_output, question_input],
            )
            
            question_input.submit(
                fn=respond,
                inputs=[question_input, chatbot, top_k_slider, use_reranker_checkbox, initial_candidates_slider],
                outputs=[chatbot, sources_output, question_input],
            )
            
            clear_btn.click(
                fn=lambda: ([], ""),
                outputs=[chatbot, sources_output],
            )
            
            # Examples
            gr.Markdown("### 💡 Ví dụ câu hỏi")
            examples = gr.Examples(
                examples=[
                    "Google phát triển AI tiếng Việt như thế nào?",
                    "What are the latest AI developments?",
                    "So sánh ChatGPT và Gemini",
                    "Khi nào OpenAI ra mắt GPT-5?",
                    "AI có thể thay thế con người không?",
                ],
                inputs=question_input,
            )
        
        return interface


def create_gradio_app(
    model_name: str = "BAAI/bge-m3",
    device: str = "cuda",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    api_key: Optional[str] = None,
    use_reranker: bool = True,
    reranker_model: str = "BAAI/bge-reranker-base",
    ollama_model: str = "qwen2.5:7b",
    collection: str = "vnexpress_news",
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
):
    """
    Create and launch Gradio app.
    
    Args:
        model_name: Embedding model name
        device: Device for models ('cuda' or 'cpu')
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        api_key: Optional Qdrant API key
        use_reranker: Whether to use reranker
        reranker_model: Reranker model name
        ollama_model: Ollama model name
        collection: Qdrant collection name
        server_name: Server host (127.0.0.1 for localhost, 0.0.0.0 for network access)
        server_port: Server port
        share: Whether to create public share link
    """
    ui = RAGChatbotUI(
        model_name=model_name,
        device=device,
        qdrant_host=qdrant_host,
        qdrant_port=qdrant_port,
        api_key=api_key,
        use_reranker=use_reranker,
        reranker_model=reranker_model,
        ollama_model=ollama_model,
        collection=collection,
    )
    
    interface = ui.create_interface()
    
    # Determine display URL
    if server_name == "0.0.0.0":
        display_url = f"http://localhost:{server_port}"
        network_url = f"http://[your-ip]:{server_port}"
        print(f"Starting Gradio server...")
        print(f"Local access: {display_url}")
        print(f"Network access: {network_url}")
    else:
        display_url = f"http://{server_name}:{server_port}"
        print(f"Starting Gradio server on {display_url}")
    
    print("Press Ctrl+C to stop the server\n")
    
    try:
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True,
            inbrowser=False,  # Don't auto-open browser
        )
    except Exception as e:
        print(f"Error starting server: {e}")
        print(f"\n Try these solutions:")
        print(f"   1. Use different port: --server-port 7861")
        print(f"   2. Use localhost: --server-name 127.0.0.1")
        print(f"   3. Check if port {server_port} is already in use")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch RAG News Chatbot Gradio UI")
    parser.add_argument("--model-name", default="BAAI/bge-m3", help="Embedding model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--api-key", default=None, help="Qdrant API key")
    parser.add_argument("--use-reranker", action="store_true", default=True, help="Use reranker")
    parser.add_argument("--no-reranker", dest="use_reranker", action="store_false", help="Disable reranker")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-base", help="Reranker model")
    parser.add_argument("--ollama-model", default="qwen2.5:7b", help="Ollama model name")
    parser.add_argument("--collection", default="vnexpress_news", help="Qdrant collection name")
    parser.add_argument("--server-name", default="127.0.0.1", help="Server host (127.0.0.1 for localhost, 0.0.0.0 for network)")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    
    args = parser.parse_args()
    
    create_gradio_app(
        model_name=args.model_name,
        device=args.device,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        api_key=args.api_key,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        ollama_model=args.ollama_model,
        collection=args.collection,
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )

