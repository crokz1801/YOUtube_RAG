## 🎬 YouTube Transcript RAG Q&A System

This project demonstrates a simple implementation of a **Retrieval-Augmented Generation (RAG)** pipeline to answer questions based on the transcript of a specific YouTube video.

The system uses popular Python libraries like `youtube-transcript-api`, `langchain`, and `huggingface-embeddings` to extract text, process it, create a vector database, and then use a Large Language Model (LLM) to answer queries.

### ⚙️ Procedure Overview

The RAG pipeline is executed in four main steps: **Data Ingestion**, **Indexing**, **Retrieval**, and **Augmentation/Generation**.



---

### 1. Data Ingestion (Getting the Transcript)

The process begins by fetching the raw text content of the YouTube video.

* **Tool Used:** `youtube-transcript-api`
* **Process:** The script targets a specific `video_id` (e.g., `Gfr50f6ZBvo`) and fetches the available English transcript. The structured transcript snippets (with timestamps) are then flattened into a single, continuous string of text.

### 2. Indexing and Chunking

The raw text data is too large for efficient processing, so it is split into smaller, manageable chunks and indexed into a vector database.

* **Chunking:** `RecursiveCharacterTextSplitter` is used to divide the long transcript into smaller `chunks` of text (e.g., 1000 characters with 200 characters overlap).
* **Embedding:** The `HuggingFaceEmbeddings` model (`all-MiniLM-L6-v2`) is used to convert each text chunk into a dense numerical vector (embedding). These embeddings capture the semantic meaning of the text.
* **Vector Store:** The chunks and their embeddings are stored in a **FAISS** (Facebook AI Similarity Search) index, which is optimized for fast similarity searches.

### 3. Retrieval

When a user asks a question, the system retrieves the most relevant chunks from the FAISS vector store.

* **Query Embedding:** The user's `question` is also converted into an embedding using the same model.
* **Similarity Search:** The system performs a similarity search in the FAISS index to find the $k$ (e.g., $k=4$) chunks whose embeddings are closest to the question's embedding. These retrieved chunks form the **context**.

### 4. Augmentation and Generation

The retrieved context is packaged with the original question and a specific instruction (Prompt Template) to guide the LLM's answer generation.

* **Prompt Template:** A `PromptTemplate` instructs the LLM to act as a helpful assistant and **answer ONLY from the provided context**, ensuring the response is grounded in the source material.
    ```python
    template = """
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """
    ```
* **LLM (Generation):** A local **Ollama** instance running the `llama3` model is used to process the final prompt and generate the response. The LLM synthesizes the information from the provided context to form a coherent answer.

---

### 💻 Setup and Execution

#### Dependencies

Ensure you have Python installed, then install the necessary libraries:

```bash
pip install -q youtube-transcript-api langchain-community langchain-openai \
faiss-cpu tiktoken python-dotenv langchain_huggingface langchain langchain-ollama
