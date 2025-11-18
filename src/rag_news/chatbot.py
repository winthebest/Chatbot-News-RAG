import ollama
from typing import List, Dict

from .retriever import init_rag_components, retrieve_news


def build_prompt(question: str, contexts: List[Dict]) -> str:
    """
    Build prompt for LLM: includes context (from Qdrant) + question.
    """
    if not contexts:
        context_text = "Không có ngữ cảnh nào phù hợp trong cơ sở dữ liệu."
    else:
        chunks = []
        for i, ctx in enumerate(contexts, start=1):
            score = ctx.get("score", 0.0)
            vector_score = ctx.get("vector_score")
            rerank_score = ctx.get("rerank_score")
            lang = ctx.get("lang", "?")
            
            # Create prefix with appropriate score
            if rerank_score is not None and vector_score is not None:
                prefix = f"[doc {i} | rerank_score={rerank_score:.4f} | vector_score={vector_score:.4f} | lang={lang}]"
            elif vector_score is not None:
                prefix = f"[doc {i} | vector_score={vector_score:.4f} | lang={lang}]"
            else:
                prefix = f"[doc {i} | score={score:.4f} | lang={lang}]"
            
            text = ctx.get("text") or ""
            chunks.append(f"{prefix}\n{text}")
        context_text = "\n\n---\n\n".join(chunks)

    prompt = f"""Bạn là trợ lý AI chuyên tóm tắt và trả lời câu hỏi dựa trên các bài báo từ VNExpress.

Ngữ cảnh (các đoạn tin tức liên quan):

{context_text}

---

Yêu cầu:
- Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu.
- Nếu câu hỏi bằng tiếng Anh thì trả lời bằng tiếng Anh.
- Chỉ sử dụng thông tin trong ngữ cảnh ở trên.
- Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói: "Trong dữ liệu hiện tại, tôi không tìm thấy thông tin đủ rõ để trả lời."
- Nếu có thể, hãy nhắc lại tiêu đề hoặc mô tả ngắn về bài báo liên quan.

Câu hỏi của người dùng:
{question}

Câu trả lời:
"""
    return prompt


def answer_with_ollama(prompt: str, model_name: str = "qwen2.5:7b") -> str:
    """
    Call local model via Ollama to generate answer.
    """
    resp = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Bạn là trợ lý AI trả lời dựa trên ngữ cảnh được cung cấp. Không bịa thêm thông tin ngoài ngữ cảnh.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return resp["message"]["content"].strip()


def main():
    # 1) Initialize embedding model + reranker + Qdrant client
    model, reranker, client = init_rag_components(
        model_name="BAAI/bge-m3",
        device="cuda",          # Change to "cpu" if CUDA error occurs
        qdrant_host="localhost",
        qdrant_port=6333,
        api_key=None,
        use_reranker=True,      # Enable re-ranking to improve quality
        reranker_model="BAAI/bge-reranker-base",
    )

    print("=== Chatbot RAG News (VNExpress + Ollama) ===")
    print("Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:
        question = input("Bạn: ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Tạm biệt!")
            break

        # 2) Retrieve context from Qdrant (with re-ranking)
        contexts = retrieve_news(
            client=client,
            model=model,
            question=question,
            collection="vnexpress_news",
            top_k=5,
            reranker=reranker,  # Use reranker to improve results
            initial_candidates=20,  # Fetch 20 candidates, rerank, then select top 5
        )

        # 3) Build prompt for LLM
        prompt = build_prompt(question, contexts)

        # 4) Call local LLM via Ollama
        answer = answer_with_ollama(prompt, model_name="qwen2.5:7b")

        print("\nChatbot:", answer)

        if contexts:  # Only print when documents are available
            print("\n" + "="*80)
            print("Reference Sources (with scores):")
            print("="*80)
            for i, ctx in enumerate(contexts, start=1):
                title = ctx.get("title") or "(no title)"
                url = ctx.get("url") or "(no url)"
                score = ctx.get("score", 0.0)
                vector_score = ctx.get("vector_score")
                rerank_score = ctx.get("rerank_score")
                lang = ctx.get("lang", "?")
                
                # Display appropriate score
                if rerank_score is not None and vector_score is not None:
                    # Has both rerank and vector score
                    print(f"\n[{i}] {title}")
                    print(f"    URL: {url}")
                    print(f"    Rerank Score: {rerank_score:.4f} | Vector Score: {vector_score:.4f} | Lang: {lang}")
                elif vector_score is not None:
                    # Only vector score (no rerank)
                    print(f"\n[{i}] {title}")
                    print(f"    URL: {url}")
                    print(f"    Vector Score: {vector_score:.4f} | Lang: {lang}")
                else:
                    # Only general score
                    print(f"\n[{i}] {title}")
                    print(f"    URL: {url}")
                    print(f"    Score: {score:.4f} | Lang: {lang}")
            print("="*80)


if __name__ == "__main__":
    main()

