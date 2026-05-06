from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import io

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 모델 & token2idx 로드
model = tf.keras.models.load_model("model/stealer_model.keras")
with open("token2idx.pkl", "rb") as f:
    token2idx = pickle.load(f)


# smart_encode 함수 (Colab과 동일하게)
def smart_encode(val, mapping, max_len=None):
    if isinstance(val, str):
        tokens = val.strip().split()
        encoded = [mapping.get(t, 0) for t in tokens]
        if max_len:
            encoded = encoded[:max_len] + [0] * (max_len - len(encoded))
        return encoded
    return mapping.get(str(val), 0)


def preprocess(df: pd.DataFrame):
    # ⚠️ Colab 전처리 로직을 여기에 그대로 복붙하세요
    # 예시:
    X = []
    for _, row in df.iterrows():
        encoded_row = smart_encode(row["your_column"], token2idx)
        X.append(encoded_row)
    return np.array(X)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    X = preprocess(df)
    preds = model.predict(X)
    scores = (preds[:, 1] * 100).tolist()  # 위험 점수 (%)

    results = []
    for i, score in enumerate(scores):
        results.append({
            "index": i,
            "risk_score": round(score, 2),
            "label": "위험" if score >= 50 else "정상"
        })

    return {"results": results}
