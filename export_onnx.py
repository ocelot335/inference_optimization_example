from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction

model_id = "sergeyzh/rubert-mini-frida"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)

tokenizer.save_pretrained("onnx_model/")
model.save_pretrained("onnx_model/")
