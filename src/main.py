from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.query import router as query_router
from src.retriever.retriever import Retriever

app = FastAPI(
    title="ML API",
    description="API for ML Model Inference",
    version="1.0.0",
)

# Initialize the retriever (it will automatically load the data)
retriever = Retriever()

@app.get("/")
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Include the query router
app.include_router(query_router, prefix="/api")
