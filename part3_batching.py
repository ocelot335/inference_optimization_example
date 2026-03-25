import asyncio
import time
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

MAX_BATCH_SIZE = 32
MAX_BATCH_WINDOW = 0.01  # 10 миллисекунд

queue = asyncio.Queue()
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


async def batch_worker():
    while True:
        batch = []
        item = await queue.get()
        batch.append(item)

        start_time = time.time()
        while len(batch) < MAX_BATCH_SIZE:
            elapsed = time.time() - start_time
            if elapsed >= MAX_BATCH_WINDOW:
                break
            try:
                timeout = MAX_BATCH_WINDOW - elapsed
                item = await asyncio.wait_for(queue.get(), timeout=timeout)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        try:
            texts = [req["text"] for req in batch]
            inputs = tokenizer(
                texts, padding=True, truncation=True, return_tensors="np"
            )

            outputs = model(**inputs)
            embeddings = mean_pooling(outputs, inputs["attention_mask"])

            for i, req in enumerate(batch):
                req["future"].set_result(embeddings[i].tolist())

        except Exception as e:
            for req in batch:
                if not req["future"].done():
                    req["future"].set_exception(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained("onnx_model/")
    model = ORTModelForFeatureExtraction.from_pretrained("onnx_model/")

    worker_task = asyncio.create_task(batch_worker())
    yield
    worker_task.cancel()


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(request: TextRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    await queue.put({"text": request.text, "future": future})
    embedding = await future
    return {"embedding": embedding}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("part3_batching:app", host="0.0.0.0", port=8567, reload=False)
