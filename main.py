from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="SLY Bid Prediction")

bundle = joblib.load('model.pkl')
model = bundle['model']
le_업종 = bundle['le_업종']
le_발주처 = bundle['le_발주처']
le_키워드 = bundle['le_키워드']

class BidRequest(BaseModel):
    업종: str
    금액: float
    경쟁자수: int
    발주처: str
    키워드: str

@app.post("/predict")
def predict(req: BidRequest):
    try:
        업종_enc = int(le_업종.transform([req.업종])[0])
        발주처_enc = int(le_발주처.transform([req.발주처])[0])
        키워드_enc = int(le_키워드.transform([req.키워드])[0])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"알수없는 카테고리: {e}")

    x = np.array([[업종_enc, req.금액, req.경쟁자수, 발주처_enc, 키워드_enc]])
    rate = model.predict(x)[0]
    return {"predicted_rate": round(float(rate), 5)}