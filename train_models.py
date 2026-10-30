"""
모델 학습 & joblib 저장 스크립트
실행:  python train_models.py
결과:  models/<모델키>.joblib  (성별별 모델을 dict로 묶어 저장)
       models/catalog.json     (모델 이름·설명·교차검증 점수·2040 예측)
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from height_models import BASE_YEAR, LogisticGrowth, load_long, national_mean

ROOT = Path(__file__).parent
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

nat = national_mean(load_long(ROOT / "data" / "kosis_height_raw.csv"))
cv = KFold(n_splits=5, shuffle=True, random_state=0)   # 노트북과 동일 설정

# (모델키, 이름, 설명, 학습 시작 연도, 모델 생성 함수, 노트북에 있던 모델인가)
SPECS = [
    ("baseline", "기준선 (평균값)", "연도와 상관없이 과거 평균 키를 그대로 예측하는, 아무것도 배우지 않은 모델",
     1975, lambda: DummyRegressor(strategy="mean"), False),
    ("poly1", "1차 다항회귀 (직선)", "1975~2025년 전체 추세를 직선으로 연장 — 팀이 최종 선택한 모델",
     1975, lambda: make_pipeline(PolynomialFeatures(1), LinearRegression()), True),
    ("poly2", "2차 다항회귀", "처음엔 가파르고 나중엔 완만해지는 곡선",
     1975, lambda: make_pipeline(PolynomialFeatures(2), LinearRegression()), True),
    ("poly3", "3차 다항회귀", "관측 구간 오차는 가장 작지만 2040년엔 곡선이 아래로 꺾임",
     1975, lambda: make_pipeline(PolynomialFeatures(3), LinearRegression()), True),
    ("recent", "최근 자료 직선 (2010~)", "정체기에 들어선 2010~2025년 자료만으로 그은 직선",
     2010, lambda: LinearRegression(), False),
    ("logistic", "로지스틱 성장곡선", "키가 일정 수준에서 멈춘다는 현실 지식을 반영한 S자 곡선",
     1975, lambda: LogisticGrowth(), False),
]

catalog = []
for key, name, desc, start, make, in_nb in SPECS:
    bundle, info = {}, {"key": key, "name": name, "desc": desc,
                        "train_from": start, "in_notebook": in_nb,
                        "cv_rmse": {}, "pred_2040": {}}
    for sex in ["남자", "여자"]:
        s = nat.loc[start:, sex].dropna()
        X = (s.index.values - BASE_YEAR).reshape(-1, 1).astype(float)
        y = s.values
        rmse = -cross_val_score(make(), X, y, cv=cv,
                                scoring="neg_root_mean_squared_error").mean()
        model = make().fit(X, y)
        bundle[sex] = model
        info["cv_rmse"][sex] = round(float(rmse), 2)
        info["pred_2040"][sex] = round(float(model.predict([[2040 - BASE_YEAR]])[0]), 1)
        if key == "logistic":
            info.setdefault("ceiling", {})[sex] = round(float(model.ceiling_), 1)
    joblib.dump(bundle, OUT / f"{key}.joblib", compress=3)
    catalog.append(info)
    print(f"{name:<22} RMSE 남 {info['cv_rmse']['남자']:.2f} / 여 {info['cv_rmse']['여자']:.2f}"
          f"  | 2040 남 {info['pred_2040']['남자']} / 여 {info['pred_2040']['여자']}")

(OUT / "catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
print("저장 완료 →", sorted(p.name for p in OUT.iterdir()))
