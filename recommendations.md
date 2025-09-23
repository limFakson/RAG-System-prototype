# 📌 RAG System – Improvement Notes
1. Chunking Strategy
    - Increase CHUNK_SIZE → 1000 (from 600).
    - Reduce CHUNK_OVERLAP → ~150.
    - Larger chunks give GPT more context per match.

2. Retrieval Depth
    - Raise k in ask() → 8–10 instead of 5.
    - Ensures more candidate passages are considered.

3. Fallbacks for Weak Retrieval
    - After Pinecone query, check similarity scores.
    - If all results < 0.75 (example threshold):
    - Pass the raw query directly into GPT.

Add disclaimer:
    
    *“⚠️ This answer may not be based on the ingested documents. Here’s what I know from general knowledge: …”*

4. Fine-tuning Options
    - Fine-tune embeddings → improve retrieval on domain-specific jargon.
    - Fine-tune chat model on Q&A pairs from client docs → domain answers become sharper, reduce hallucination.
    - Especially useful if docs are technical, regulatory, or legal.

5. UI/UX Improvements
    - Execute query automatically when user presses Enter.
    - Provide a clearer “Sources” section in Streamlit app.
    - Show retrieval scores (optional) for transparency.

✅ These fixes can be implemented in stages:
- Chunking & retrieval depth → quick win.
- Fallback logic → medium effort.
- Fine-tuning → long-term client trust improvement.