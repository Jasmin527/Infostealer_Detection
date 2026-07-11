from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import io
import uvicorn
import re

from pathlib import Path

# =====================================================
# FastAPI
# =====================================================

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# =====================================================
# 모델 로드
# =====================================================

model = tf.keras.models.load_model(
    BASE_DIR / "model" / "stealer_model.keras"
)

with open(BASE_DIR / "token2idx.pkl", "rb") as f:
    token2idx = pickle.load(f)

# =====================================================
# 학습 코드와 동일한 전처리
# =====================================================

MAX_LEN = 3000
CHUNK = MAX_LEN // 3


def tokenize(text):
    text = text.lower()

    tokens = re.split(
        r'[\s\x00-\x1f\x7f-\xff\\/,;:|<>{}()\[\]"\'=+*&^%$#@!?]+',
        text
    )

    return [
        t for t in tokens
        if 2 <= len(t) <= 50
    ]


def smart_encode(text):

    tokens = tokenize(str(text))

    n = len(tokens)

    if n <= MAX_LEN:
        selected = tokens

    else:
        front = tokens[:CHUNK]

        mid = tokens[
            n // 2 - CHUNK // 2:
            n // 2 + CHUNK // 2
        ]

        back = tokens[-CHUNK:]

        selected = front + mid + back

    ids = [
        token2idx.get(token, 1)
        for token in selected[:MAX_LEN]
    ]

    arr = np.zeros(MAX_LEN, dtype="int32")

    arr[:len(ids)] = ids

    return arr


def build_sequence_series(df: pd.DataFrame):
    """
    CSV로부터 학습 때와 동일한 'sequence' 텍스트를 만들어내는 함수.

    1순위: sequence / api_sequence / behavior / api_call 컬럼이 있으면 그대로 사용
    2순위: 0, 1, 2, ... 처럼 숫자로 된 컬럼들(wide format)이 있으면
           행 단위로 순서대로 이어붙여서 하나의 시퀀스 문자열로 재구성
    3순위: 위 두 경우 다 아니면 첫 번째 컬럼을 사용 (경고 출력)
    """

    possible_cols = [
        "sequence",
        "api_sequence",
        "behavior",
        "api_call"
    ]

    for col in possible_cols:
        if col in df.columns:
            print(f"[INFO] 사용 컬럼: {col}")
            return df[col]

    # 숫자로만 이루어진 컬럼명(0, 1, 2, ...)을 순서대로 정렬해서 추출
    numeric_cols = sorted(
        [c for c in df.columns if str(c).isdigit()],
        key=lambda x: int(x)
    )

    if numeric_cols:
        print(
            f"[INFO] 'sequence' 계열 컬럼 없음 → "
            f"숫자 컬럼 {len(numeric_cols)}개를 시퀀스로 결합 "
            f"({numeric_cols[0]}~{numeric_cols[-1]})"
        )

        def row_to_sequence(row):
            vals = row[numeric_cols].dropna().astype(str).tolist()
            return " ".join(vals)

        return df.apply(row_to_sequence, axis=1)

    # 마지막 fallback: 첫 번째 컬럼 (여기로 오면 결과를 신뢰하면 안 됨)
    fallback_col = df.columns[0]
    print(
        f"[WARNING] sequence 계열/숫자 컬럼을 찾지 못함 → "
        f"'{fallback_col}' 컬럼을 임시로 사용합니다. "
        f"결과가 모두 동일하게 나온다면 이 컬럼이 원인일 수 있습니다."
    )
    return df[fallback_col]


def preprocess(df: pd.DataFrame):

    seq_series = build_sequence_series(df)

    X = np.zeros(
        (len(df), MAX_LEN),
        dtype="int32"
    )

    for i, seq in enumerate(seq_series):
        X[i] = smart_encode(seq)

    return X


# =====================================================
# 페이지
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# =====================================================
# 예측 API
# =====================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

        print("[INFO] CSV 컬럼:")
        print(df.columns.tolist())

        X = preprocess(df)

        preds = model.predict(
            X,
            verbose=0
        )

        scores = (
            preds.flatten() * 100
        ).tolist()

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

    except Exception as e:

        print("[ERROR]", str(e))

        return {
            "error": str(e)
        }


# =====================================================
# 실행
# =====================================================

if __name__ == "__main__":

    uvicorn.run(
        "stealer_service.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )