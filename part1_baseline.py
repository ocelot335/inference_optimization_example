from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

model = None


class TextRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = SentenceTransformer("sergeyzh/rubert-mini-frida", device="cpu")
    yield
    model = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: TextRequest):
    embedding = model.encode(request.text).tolist()
    return {"embedding": embedding}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("part1_baseline:app", host="0.0.0.0", port=8567, reload=False)
