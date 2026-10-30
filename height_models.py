"""
공통 모듈: 데이터 전처리 + 로지스틱 성장곡선 모델
- train_models.py(모델 학습)와 app.py(웹앱)가 함께 import 합니다.
- joblib로 저장한 사용자 정의 모델(LogisticGrowth)을 불러오려면
  이 파일이 app.py와 같은 폴더에 있어야 합니다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin

BASE_YEAR = 2000  # 연도에서 2000을 빼서 사용 (노트북과 동일)


def load_long(csv_path: Path) -> pd.DataFrame:
    """KOSIS 가로형 CSV(cp949) → 세로형(연도·시도·성별·키) DataFrame"""
    raw = pd.read_csv(csv_path, encoding="cp949", header=None)
    hdr, vals = raw.iloc[:3, 2:], raw.iloc[3, 2:]
    return pd.DataFrame({
        "연도": hdr.iloc[0].astype(int).values,
        "시도": hdr.iloc[1].values,
        "성별": hdr.iloc[2].values,
        "키": pd.to_numeric(vals.values, errors="coerce"),
    })


def national_mean(long: pd.DataFrame) -> pd.DataFrame:
    """결측 제거 후 연도·성별별 시도 단순평균 (노트북 1-2 단계)"""
    return (long.dropna(subset=["키"])
                .groupby(["연도", "성별"])["키"].mean()
                .unstack().round(2))


def _logistic(x, L, k, x0, b):
    return b + L / (1 + np.exp(-k * (x - x0)))


class LogisticGrowth(BaseEstimator, RegressorMixin):
    """키가 어느 수준에서 멈추는(포화하는) S자 성장곡선
    y = b + L / (1 + exp(-k (x - x0)))
    b: 출발 키, b+L: 최종적으로 수렴하는 키(상한)"""

    def fit(self, X, y):
        x = np.asarray(X, dtype=float).ravel()
        y = np.asarray(y, dtype=float)
        p0 = [y.max() - y.min() + 1, 0.1, float(np.median(x)), y.min() - 0.5]
        self.params_, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=20000)
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float).ravel()
        return _logistic(x, *self.params_)

    @property
    def ceiling_(self):
        return self.params_[3] + self.params_[0]
