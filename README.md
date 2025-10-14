# Automating Retrieval-Augmented Q&A with AI

Learn how to build a **RAG powered assistant** that automatically retrieves, grounds, and answers user questions using **local PDFs**, **image embeddings**, and a **Phi 3 Vision LLM**, all running on **CPU** with an in memory **Chroma vector store**.

Let’s dive in!

---

## Core Capabilities

* **Unified Document Intelligence**
  Seamlessly ingests PDFs, extracts text and images, and stores their embeddings in an in memory vector store without any external database.

* **Lightweight Embedding Engine**
  Uses **Jina CLIP V1** to generate text and image embeddings for efficient similarity search.

* **Smart Retrieval Pipeline**

  1. Top k semantic vector hits
  2. Automatic similarity normalization and ranking
  3. Merges image and text results into a unified context

* **Concise, Context Aware Answers**
  Powered by **Phi 3 Vision 128k Instruct**, it generates clear, concise, and context grounded responses.

  * Up to 3 sentences
  * Based only on retrieved context
  * Safe fallback: *“I don’t know based on the provided information.”*

* **No External Dependencies**
  Works offline, with no API keys, GPUs, or web connectors required.
  Everything runs locally using only the allowed libraries:
  `torch`, `chromadb`, `numpy`, `io`, `fitz`, `requests`, `PIL`, `transformers`.

---

## What You’ll Do

1. **Instantiate the Agent**

   ```python
   rag_agent = RagAgent(
       model="microsoft/phi-3-vision-128k-instruct",
       embedding_model="jinaai/jina-clip-v1"
   )
   ```

2. **Index a Document**

   ```python
   rag_agent.index_document("https://arxiv.org/pdf/2407.12345.pdf")
   ```

3. **Ask Questions**

   ```python
   response = rag_agent.invoke("What problem does the paper tackle?")
   print(response["answer"])
   ```

4. **Inspect Results**
   View top retrieved snippets, similarity scores, and which document pages informed the answer:

   ```python
   for hit in response["hits"][:3]:
       print(hit["score"], hit["metadata"], hit["document"][:150])
   ```

---

## Optional Enhancements

* Add **multi document memory** by indexing multiple PDFs with `name="project_a"`, `name="project_b"`, etc.
* Integrate **web enrichment** by adding a lightweight search fallback such as Tavily or SerpAPI.
* Visualize **retrieval flow** in Jupyter using a Mermaid or LangGraph diagram.
* Log question and answer sessions to JSON for reproducibility or audit trails.

---

## Summary

This notebook demonstrates a **minimal but complete Retrieval Augmented Generation (RAG)** setup from ingestion to reasoning using **only open source CPU friendly tools**.
You will understand how every stage works: ingestion -> embedding -> retrieval -> grounded generation.

---
