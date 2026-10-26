# Module 3 — GenAI Support Assistant: Grounded RAG & LangGraph Service

This module implements a production-grade, offline-first GenAI retrieval-augmented generation (RAG) support service for Zepto, orchestrated with **LangGraph**, backed by **Sentence Transformers** and **ChromaDB**, validated via **Pydantic**, served using **FastAPI**, and containerized with **Docker**.

The required graded baseline runs in **deterministic offline mock mode** (`MOCK_LLM=1` by default) requiring zero API keys, zero signups, and zero external network calls.

---

## RAG Pipeline Architecture & Data Flow

```
                                USER QUERY
                                    │
                                    ▼
                          [Node: classify_intent]
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            (policy_question)               (general_question)
                    │                               │
                    ▼                               ▼
       [Node: retrieve_and_answer]        [Node: direct_answer]
                    │                               │
        ┌───────────┴───────────┐                   │
        │ all-MiniLM-L6-v2      │                   │
        │ Query Embedding       │                   │
        │           │           │                   │
        │           ▼           │                   │
        │ ChromaDB Top-3 Chunks │                   │
        └───────────┬───────────┘                   │
                    │                               │
       MOCK_LLM=1: Canned Template     MOCK_LLM=1: Canned String
       MOCK_LLM=0: Real LLM Prompt     MOCK_LLM=0: Real LLM Direct
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                         [Pydantic Response]
                                    │
                                    ▼
                         FastAPI (POST /ask)
                                    │
                                    ▼
                            Docker Container
```

### Detailed Pipeline Stages & Component Roles
1. **Ingestion (`support_assistant/ingest.py`)**:
   - Reads 8 policy documents from [`support_assistant/documents/`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents).
   - Generates 384-dimensional dense vector embeddings using the open-source `sentence-transformers/all-MiniLM-L6-v2` model.
   - Stores documents, vector embeddings, and metadata (`source: filename`) into a persistent ChromaDB collection named `zepto_policies` under `support_assistant/chroma_db/`.
2. **Intent Classification (`support_assistant/graph.py` -> `classify_intent`)**:
   - In default mock mode (`MOCK_LLM=1`), classifies queries using a deterministic keyword heuristic:
     - Keywords: `"delivery"`, `"return"`, `"refund"`, `"membership"`, `"tracking"`, `"cancel"`, `"gift card"`, `"support hours"`.
     - Matching queries $\rightarrow$ `policy_question`.
     - Non-matching queries $\rightarrow$ `general_question`.
   - In optional real mode (`MOCK_LLM=0`), invokes an LLM classifier.
3. **Retrieval (`support_assistant/graph.py` -> `retrieve_and_answer`)**:
   - Embeds the user query via `all-MiniLM-L6-v2`.
   - Performs cosine similarity search over `zepto_policies` in ChromaDB, retrieving the top-3 most relevant document chunks.
   - *Note*: Vector retrieval runs for real in both mock and real LLM modes.
4. **Answer Generation**:
   - **Policy Question**:
     - *Mock mode (`MOCK_LLM=1`)*: Generates deterministic answer: `f"Based on the retrieved context: {top_chunk[:200]}"`.
     - *Real mode (`MOCK_LLM=0`)*: Formats context and query into the structured prompt template and invokes the LLM.
   - **General Question (`direct_answer`)**:
     - *Mock mode (`MOCK_LLM=1`)*: Returns fixed string: `"I can only answer questions about Zepto policies right now."` with empty sources.
     - *Real mode (`MOCK_LLM=0`)*: Prompts the LLM directly without retrieval.
5. **Validation & Delivery (`support_assistant/main.py`)**:
   - Validates outputs against Pydantic model `SupportResponse` (`answer`, `sources`, `confidence`).
   - Served via FastAPI endpoint `POST /ask`.

---

## Corpus Documents

Located in [`support_assistant/documents/`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents):
- [`doc_01_delivery.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_01_delivery.txt) — 10–30 min delivery, free over INR 149, INR 25 fee below threshold, INR 15 priority delivery.
- [`doc_02_returns.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_02_returns.txt) — Returns within 24h for perishables, 7 days for non-perishables, wallet/original refund.
- [`doc_03_membership.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_03_membership.txt) — Basic (free), Zepto Pass (INR 49/mo), Zepto Pass+ (INR 99/mo).
- [`doc_04_order_tracking.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_04_order_tracking.txt) — Live rider-tracking map, support contact after 20 minutes past ETA.
- [`doc_05_cancellation.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_05_cancellation.txt) — Free cancellation before 'Packed' status (first 2 minutes).
- [`doc_06_damaged_missing.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_06_damaged_missing.txt) — Report within 24h, photo required if order value > INR 1000.
- [`doc_07_gift_cards.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_07_gift_cards.txt) — Fixed denominations (INR 100, 250, 500, 1000), valid for 1 year.
- [`doc_08_support_hours.txt`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/documents/doc_08_support_hours.txt) — 24/7 in-app chat (< 2 min response), email within 24h, no phone support.

---

## Structured Prompt Template

Defined in [`support_assistant/prompt.py`](file:///home/murali/Downloads/zepto-ai-ml-capstone/support_assistant/prompt.py):

```text
ROLE:
You are a Zepto customer support assistant.

CONTEXT:
Use only the policy information provided in the retrieved context.
Retrieved context:
{context}

TASK:
Answer the user's question using the provided Zepto policy context.

FORMAT:
Return a clear and helpful answer.
Mention the relevant policy information directly.

LENGTH:
Keep the answer concise and within 3 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:
User question: How can I track my order?
Context: Customers can track an active order through the Zepto application.
Answer: You can track your active order through the Zepto application.

USER QUESTION:
{question}
```

---

## Pydantic Output Schema & Retry Logic

```python
class SupportResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float
```
- In default mock mode, `sources` contains the exact retrieved filenames (e.g. `["doc_01_delivery.txt", ...]`) and `confidence = 1.0` (or `sources = []` for general questions).
- In optional real mode (`MOCK_LLM=0`), `call_llm_with_retry` automatically retries up to 2 additional times with corrective instructions if the output fails validation, before returning a graceful error fallback.

---

## FastAPI Verification & Example Call Transcripts

Run locally with default mock mode (`MOCK_LLM=1`):

### 1. Policy Question (Triggers Retrieval)
**Request**:
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the delivery fee for orders below INR 149?"}'
```

**Response**:
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": [
    "doc_01_delivery.txt",
    "doc_05_cancellation.txt",
    "doc_03_membership.txt"
  ],
  "confidence": 1.0
}
```

### 2. General Question (Bypasses Retrieval)
**Request**:
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the capital of India?"}'
```

**Response**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

---

## Containerization & Local Execution

### Local Python Run:
```bash
# 1. Ingest policy documents into ChromaDB
python support_assistant/ingest.py

# 2. Start FastAPI application
uvicorn support_assistant.main:app --host 127.0.0.1 --port 8000
```

### Docker Run:
```bash
# Build Docker container
docker build -t zepto-support-assistant .

# Run container locally
docker run -p 8000:8000 -e MOCK_LLM=1 zepto-support-assistant
```
API endpoints will be live at `http://127.0.0.1:8000` with interactive OpenAPI documentation at `http://127.0.0.1:8000/docs`.
