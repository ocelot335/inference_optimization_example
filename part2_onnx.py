from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

tokenizer = None
model = None


class TextRequest(BaseModel):
    text: str


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = np.broadcast_to(
        np.expand_dims(attention_mask, axis=-1), token_embeddings.shape
    )
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )
    return sum_embeddings / sum_mask


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("onnx_model/")
    model = ORTModelForFeatureExtraction.from_pretrained("onnx_model/")
    yield
    tokenizer = None
    model = None


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: TextRequest):
    inputs = tokenizer(
        request.text,
        padding=True,
        truncation=True,
        return_tensors="np",
    )

    outputs = model(**inputs)

    embedding = mean_pooling(outputs, inputs["attention_mask"])

    return {"embedding": embedding[0].tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("part2_onnx:app", host="0.0.0.0", port=8567, reload=False)
