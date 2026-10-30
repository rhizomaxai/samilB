# 2040년 고1 평균 키 예측 — Streamlit 웹앱 (13-4조)

## 폴더 구조
```
streamlit_app/
├── app.py               # 웹앱 (보고서 6개 탭 + 예측 체험)
├── height_models.py     # 공통 전처리 함수 + 로지스틱 모델 클래스
├── train_models.py      # 모델 학습 → models/*.joblib 저장
├── requirements.txt
├── data/kosis_height_raw.csv   # KOSIS 원본 (cp949)
└── models/
    ├── baseline.joblib  poly1.joblib  poly2.joblib  poly3.joblib
    ├── recent.joblib    logistic.joblib
    └── catalog.json     # 모델 이름·설명·CV RMSE·2040 예측
```
각 `.joblib` 파일은 `{"남자": 모델, "여자": 모델}` 딕셔너리입니다.
앱에서 모델을 고르면 해당 파일만 `joblib.load`로 불러와 바꿔 씁니다.

## 내 컴퓨터에서 실행
```bash
pip install -r requirements.txt
python train_models.py      # (선택) 모델 다시 만들기
streamlit run app.py
```

## Streamlit Community Cloud 배포
1. 이 폴더를 GitHub 저장소에 올립니다 (models 폴더 포함).
2. share.streamlit.io → New app → 저장소 선택, Main file: `app.py`.
3. `requirements.txt`의 scikit-learn 버전은 joblib을 저장한 버전과 같아야 합니다.
   버전을 바꿨다면 `python train_models.py`로 모델을 다시 저장하세요.

## 주의
- `logistic.joblib`은 `height_models.py`의 `LogisticGrowth` 클래스를 사용하므로
  `height_models.py`가 반드시 app.py와 같은 폴더에 있어야 합니다.
- 기준선·최근 직선·로지스틱 모델은 보고서의 한계를 보완하려고 웹앱에서 추가로 만든 모델입니다.
