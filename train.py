
import pandas as pd
from lightgbm import LGBMRegressor
import joblib

df = pd.read_csv("training_data.csv")

X = pd.DataFrame({
    "is_electric": (df.업종=="전기").astype(int),
    "금액": df.금액,
    "경쟁자수": df.경쟁자수,
    "발주처_hash": df.발주처.apply(lambda s: hash(s)%100),
    "키워드_hash": df.키워드.apply(lambda s: hash(s)%100),
})
y = df.사정률

model = LGBMRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
