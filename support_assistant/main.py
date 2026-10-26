from fastapi import FastAPI
from pydantic import BaseModel
from support_assistant.graph import graph, SupportResponse

app = FastAPI(title="Zepto Support Assistant")


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {"message": "Zepto Support Assistant is running"}


@app.post("/ask", response_model=SupportResponse)
def ask(request: AskRequest):
    result = graph.invoke({"query": request.query})
    return SupportResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence", 1.0)
    )