from fastapi import FastAPI
from pydantic import BaseModel
from rag_generator import RAGSystem

app = FastAPI()
rag = RAGSystem()

class QueryRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: QueryRequest):
    response = rag.generate_response(request.query)
    return {"answer": response}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
