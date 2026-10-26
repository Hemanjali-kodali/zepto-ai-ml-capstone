import os
import time
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel, Field


MOCK_LLM = os.getenv("MOCK_LLM", "1").lower() not in ("0", "false", "no")

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


class SupportState(TypedDict, total=False):
    query: str
    intent: str
    answer: str
    sources: list
    confidence: float


class SupportResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float


client = chromadb.PersistentClient(
    path="support_assistant/chroma_db"
)

collection = client.get_collection(
    name="zepto_policies"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def call_llm_with_retry(llm, prompt, max_retries=3):
    current_prompt = prompt
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(current_prompt)
            content = getattr(response, "content", str(response))
            if not content or not str(content).strip():
                raise ValueError("LLM returned empty response")
            return content.strip()
        except Exception as e:
            print(f"LLM attempt {attempt} failed: {e}")
            if attempt == max_retries:
                return "Error: Unable to generate valid response after retries."
            current_prompt = (
                f"{prompt}\n\n[Correction Notice: Attempt {attempt} failed. "
                f"You must answer strictly using the provided context and format.]"
            )
            time.sleep(1)


def get_real_llm():

    raise NotImplementedError(
        "Real LLM is optional. Set up an LLM provider before using MOCK_LLM=0."
    )


def classify_intent(state: SupportState):

    query = state["query"].lower()

    if any(keyword in query for keyword in POLICY_KEYWORDS):
        intent = "policy_question"
    else:
        intent = "general_question"

    print("\n[Node: classify_intent]")
    print("Query:", state["query"])
    print("Intent:", intent)

    return {
        "intent": intent
    }


def retrieve_and_answer(state: SupportState):

    query = state["query"]

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_documents = results["documents"][0]
    retrieved_sources = results["metadatas"][0]

    top_chunk = retrieved_documents[0]

    sources = [
        item["source"]
        for item in retrieved_sources
    ]

    if MOCK_LLM:

        answer = (
            "Based on the retrieved context: "
            + top_chunk[:200]
        )

    else:

        prompt = f"""
You are a Zepto support assistant.

Answer the user's question using only the
retrieved context below.

Do not use information that is not present
in the retrieved context.

Retrieved context:
{top_chunk}

User question:
{query}

Give a concise answer.
"""

        llm = get_real_llm()

        answer = call_llm_with_retry(
            llm,
            prompt
        )

    response = SupportResponse(
        answer=answer,
        sources=sources,
        confidence=1.0
    )

    print("\n[Node: retrieve_and_answer]")
    print("Retrieved sources:", sources)
    print("Answer:", response.answer)

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


def direct_answer(state: SupportState):

    query = state["query"]

    if MOCK_LLM:

        answer = (
            "I can only answer questions about Zepto policies right now."
        )

    else:

        prompt = f"""
You are a Zepto support assistant.

The user's question is outside the available
Zepto policy knowledge base.

Respond politely that you can only answer
questions about Zepto policies.

Do not invent Zepto policies.

User question:
{query}
"""

        llm = get_real_llm()

        answer = call_llm_with_retry(
            llm,
            prompt
        )

    response = SupportResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )

    print("\n[Node: direct_answer]")
    print("Answer:", response.answer)

    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence
    }


def route_question(state: SupportState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


graph_builder = StateGraph(SupportState)

graph_builder.add_node(
    "classify_intent",
    classify_intent
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

graph_builder.add_node(
    "direct_answer",
    direct_answer
)

graph_builder.add_edge(
    START,
    "classify_intent"
)

graph_builder.add_conditional_edges(
    "classify_intent",
    route_question,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

graph_builder.add_edge(
    "retrieve_and_answer",
    END
)

graph_builder.add_edge(
    "direct_answer",
    END
)

graph = graph_builder.compile()


if __name__ == "__main__":

    print("\n========================================")
    print("MOCK_LLM =", MOCK_LLM)
    print("========================================")

    print("\n========== POLICY QUESTION ==========")

    policy_result = graph.invoke({
        "query": "How can I track my delivery?"
    })

    print("\nFinal response:")
    print(policy_result)

    print("\n========== GENERAL QUESTION ==========")

    general_result = graph.invoke({
        "query": "What is the capital of India?"
    })

    print("\nFinal response:")
    print(general_result)