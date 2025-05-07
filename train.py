import pandas as pd
from lightgbm import LGBMRegressor
import joblib
from sklearn.preprocessing import LabelEncoder

def train():
    df = pd.read_csv('training_data.csv')
    le_업종 = LabelEncoder().fit(df['업종'])
    le_발주처 = LabelEncoder().fit(df['발주처'])
    le_키워드 = LabelEncoder().fit(df['키워드'])

    df['업종_enc'] = le_업종.transform(df['업종'])
    df['발주처_enc'] = le_발주처.transform(df['발주처'])
    df['키워드_enc'] = le_키워드.transform(df['키워드'])

    feature_cols = ['업종_enc', '금액', '경쟁자수', '발주처_enc', '키워드_enc']
    X = df[feature_cols]
    y = df['사정률']

    model = LGBMRegressor(n_estimators=200)
    model.fit(X, y)

    bundle = {
        'model': model,
        'le_업종': le_업종,
        'le_발주처': le_발주처,
        'le_키워드': le_키워드
    }
    joblib.dump(bundle, 'model.pkl')
    print("✅ 모델과 인코더를 model.pkl로 저장했습니다.")

if __name__ == '__main__':
    train()