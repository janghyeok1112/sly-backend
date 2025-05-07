
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

class BidRequest(BaseModel):
    업종: str
    금액: float
    경쟁자수: int
    발주처: str
    키워드: str

@app.post("/predict")
def predict(req: BidRequest):
    X = [[
        1 if req.업종=="전기" else 0,
        req.금액,
        req.경쟁자수,
        hash(req.발주처)%100,
        hash(req.키워드)%100
    ]]
    rate = model.predict(X)[0]
    return {"predicted_rate": round(float(rate), 5)}
