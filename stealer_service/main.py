from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import io
import uvicorn
from pathlib import Path

app = FastAPI()

# 현재 main.py가 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# templates 폴더 설정
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# 모델 & token2idx 로드
model = tf.keras.models.load_model(
    BASE_DIR / "model" / "stealer_model.keras"
)

with open(BASE_DIR / "token2idx.pkl", "rb") as f:
    token2idx = pickle.load(f)


# smart_encode 함수
def smart_encode(val, mapping, max_len=None):
    if isinstance(val, str):
        tokens = val.strip().split()
        encoded = [mapping.get(t, 0) for t in tokens]

        if max_len:
            encoded = encoded[:max_len] + [0] * max(0, max_len - len(encoded))

        return encoded

    return mapping.get(str(val), 0)


# 전처리 함수
def preprocess(df: pd.DataFrame):
    X = []

    for _, row in df.iterrows():
        # TODO:
        # "your_column"을 실제 CSV 컬럼명으로 변경해야 함
        encoded_row = smart_encode(
            row["your_column"],
            token2idx
        )

        X.append(encoded_row)

    return np.array(X)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    df = pd.read_csv(
        io.BytesIO(contents)
    )

    X = preprocess(df)

    preds = model.predict(X)

    scores = (preds[:, 1] * 100).tolist()

    results = []

    for i, score in enumerate(scores):
        results.append({
            "index": i,
            "risk_score": round(score, 2),
            "label": "위험" if score >= 50 else "정상"
        })

    return {
        "results": results
    }

if __name__ == "__main__":
    uvicorn.run(
        "stealer_service.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )