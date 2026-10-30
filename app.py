"""
2040년 고1 평균 키는 몇 cm일까? — 삼일고 2학년 2반 13-4조 데이터분석 보고서 웹앱
실행:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from height_models import BASE_YEAR, load_long, national_mean  # noqa: F401 (LogisticGrowth 로드에 필요)

ROOT = Path(__file__).parent
SEX_COLOR = {"남자": "#2563eb", "여자": "#e11d48"}
MODEL_COLOR = {"baseline": "#94a3b8", "poly1": "#f59e0b", "poly2": "#10b981",
               "poly3": "#8b5cf6", "recent": "#0ea5e9", "logistic": "#ef4444"}

st.set_page_config(page_title="2040년 고1 평균 키 예측", page_icon="📏", layout="wide")


# ───────────────────────── 데이터·모델 로드 ─────────────────────────
@st.cache_data
def get_data():
    long = load_long(ROOT / "data" / "kosis_height_raw.csv")
    return long, national_mean(long)


@st.cache_data
def get_catalog():
    return json.loads((ROOT / "models" / "catalog.json").read_text(encoding="utf-8"))


@st.cache_resource          # 모델별로 한 번만 joblib 로드해 메모리에 보관
def load_model(key: str):
    return joblib.load(ROOT / "models" / f"{key}.joblib")


def predict(key, sex, years):
    x = (np.asarray(years, dtype=float) - BASE_YEAR).reshape(-1, 1)
    return load_model(key)[sex].predict(x)


long, nat = get_data()
catalog = get_catalog()
cat_by_key = {c["key"]: c for c in catalog}
name_by_key = {c["key"]: c["name"] for c in catalog}

# ───────────────────────── 사이드바 ─────────────────────────
with st.sidebar:
    st.markdown("### 📏 13-4조 데이터분석")
    st.caption("삼일고등학교 · 2학년 2반")
    st.markdown("""
| 이름 | 역할 |
|---|---|
| 이정원 | 팀장 · 발표 |
| 채성현 | 데이터 수집 · 전처리 |
| 윤봄이 | 모델링 · 평가 |
| 송아랑 | 보고서 · 시각화 |
""")
    st.divider()
    st.caption("데이터: KOSIS 국가통계포털 「시·도별 학생표본 신체(키) 검사 현황」 1975~2025, 공공누리")
    st.caption("제출일: 2026. 10. 29.")

# ───────────────────────── 헤더 ─────────────────────────
st.title("📏 2040년 고1 평균 키는 몇 cm일까?")
st.markdown("**KOSIS 학생건강검사 데이터로 17세 고등학생의 미래 평균 키 예측하기**")

c1, c2, c3, c4 = st.columns(4)
c1.metric("데이터 크기", f"{len(long):,}행", help="연도·시도·성별·키 4열, 1975~2025 (2020 제외)")
c2.metric("2025년 실제 평균 (남)", f"{nat.loc[2025, '남자']:.1f} cm")
c3.metric("2025년 실제 평균 (여)", f"{nat.loc[2025, '여자']:.1f} cm")
c4.metric("팀 최종 모델 2040 예측 (남/여)",
          f"{cat_by_key['poly1']['pred_2040']['남자']} / {cat_by_key['poly1']['pred_2040']['여자']}",
          help="1차 다항회귀(직선) 모델, 단위 cm")

tabs = st.tabs(["① 문제 정의", "② 데이터", "③ 탐색", "④ 모델 평가", "⑤ 2040 예측 체험", "⑥ 결론·한계"])

# ───────────────────────── ① 문제 정의 ─────────────────────────
with tabs[0]:
    st.subheader("요약")
    st.info("우리나라 17세 고등학생의 평균 키가 2040년에 어느 정도일지 알아보기 위해 KOSIS 학생건강검사 "
            "표본통계(1975~2025)를 분석했다. 연도를 단서로 평균 키를 맞히는 회귀 모델(1·2·3차 다항회귀 등)을 "
            "만들어 비교했고, 모델에 따라 2040년 예측값이 크게 달라진다는 것을 확인했다.")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("배경과 목적")
        st.write("과거부터 현재까지 고등학생의 평균 키가 어떻게 변해 왔는지 궁금했고, 이 추세로 2040년의 평균 키를 "
                 "예측해 보고 싶었다. 평균 키의 변화를 보면 시대에 따른 청소년 성장 추세를 알 수 있고, 통계 자료로 "
                 "미래를 예측하는 데이터 분석의 의미도 이해할 수 있다.")
        st.subheader("분석 질문")
        st.markdown("""
| 단계 | 내용 |
|---|---|
| 궁금증 | 시간이 지남에 따라 고등학생들의 평균 키는 어떻게 변해 왔을까? |
| 질문 | 1975년부터 2025년까지 평균 키는 어떻게 변했을까? |
| **분석 질문** | **1975~2025년 17세 고등학생 평균 키 데이터를 바탕으로 2040년의 예상 평균 키는 몇 cm일까?** |
""")
    with right:
        st.subheader("머신러닝 문제로 바꾸기")
        st.markdown("""
| 항목 | 내용 |
|---|---|
| 정답 y | 평균 키 (cm) |
| 특성 X | 연도 − 2000 |
| 문제 종류 | ✅ **회귀** (정답이 숫자) |
| 활용 | 교육기관·청소년 성장 연구자, 데이터 분석 교육 자료 |
""")

# ───────────────────────── ② 데이터 ─────────────────────────
with tabs[1]:
    st.subheader("데이터 출처")
    st.markdown("""
| 항목 | 내용 |
|---|---|
| 데이터 이름 | 시·도별 학생표본 신체(키) 검사 현황 (고등학교 · 17세) |
| 출처 | KOSIS 국가통계포털 |
| 기간 | 1975 ~ 2025년 |
| 이용 조건 / 형식 | 공공누리 / CSV (cp949) |
""")
    st.subheader("전처리 기록")
    st.markdown("""
| 문제 | 처리 방법 | 이유 |
|---|---|---|
| 가로형 원본 (4행 × 1,492열) | 연도·시도·성별·키 4열의 세로형(1,490행)으로 변환 | 분석하기 쉬운 '한 행 = 관측값 하나' 구조 |
| 결측치 1개 (1975 경북 남자) | 해당 행 제거 | 억지로 채우지 않고 제외 |
| 2020년 자료 없음 | 채우지 않음 | 코로나19로 검사 미실시 |
| 자료형 | 키를 숫자형으로 변환 ('-' → 결측) | 계산 가능하도록 |
| 파생 변수 | 연도·성별별 17개 시·도 단순평균 → '전국' 시계열 | 연도별 추세를 한 줄로 보기 위해 |
""")
    st.subheader("시·도별로 살펴보기")
    regions = sorted(long["시도"].unique())
    pick = st.multiselect("시·도 선택", regions, default=["서울특별시", "경기도", "제주특별자치도"])
    sex_pick = st.radio("성별", ["남자", "여자"], horizontal=True, key="region_sex")
    fig = go.Figure()
    for r in pick:
        d = long[(long["시도"] == r) & (long["성별"] == sex_pick)].dropna()
        fig.add_scatter(x=d["연도"], y=d["키"], mode="lines", name=r, line=dict(width=1.5))
    fig.add_scatter(x=nat.index, y=nat[sex_pick], mode="lines", name="전국(시도 평균)",
                    line=dict(color="black", width=3))
    fig.update_layout(height=420, yaxis_title="평균 키 (cm)", xaxis_title="연도",
                      margin=dict(t=20, b=40), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("전처리된 데이터 보기 (세로형)"):
        st.dataframe(long, use_container_width=True, height=300)
        st.download_button("CSV로 내려받기", long.to_csv(index=False).encode("utf-8-sig"),
                           "height_long.csv", "text/csv")

# ───────────────────────── ③ 탐색 ─────────────────────────
with tabs[2]:
    st.subheader("17세 고등학생 평균 키 추이 (1975~2025)")
    fig = go.Figure()
    for sex in ["남자", "여자"]:
        fig.add_scatter(x=nat.index, y=nat[sex], mode="lines+markers", name=sex,
                        line=dict(color=SEX_COLOR[sex]), marker=dict(size=5))
    fig.add_vrect(x0=2005, x1=2025, fillcolor="#fde68a", opacity=0.25, line_width=0,
                  annotation_text="정체기", annotation_position="top left")
    fig.update_layout(height=420, yaxis_title="평균 키 (cm)", xaxis_title="연도", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([2, 3])
    with left:
        st.markdown("**10년 단위 평균 키 변화**")
        chk = nat.loc[[1975, 1985, 1995, 2005, 2015, 2025]].copy()
        chk["남자 증가"] = chk["남자"].diff().round(1)
        chk["여자 증가"] = chk["여자"].diff().round(1)
        st.dataframe(chk, use_container_width=True)
    with right:
        st.markdown("**탐색에서 발견한 것**")
        st.markdown("""
| | 발견 | 모델링에 주는 의미 |
|---|---|---|
| ① | 1975→2005년: 남자 약 +7cm, 여자 약 +4cm (10년에 2cm 안팎) | 과거에는 빠르게 증가 |
| ② | 2005→2025년: 남 173.60→174.21(+0.61), 여 160.98→161.57(+0.59) | 최근엔 거의 정체 → 직선 연장은 위험 |
| ③ | 2020년 조사 자료 없음 | 결측 연도는 채우지 않고 제외 |
""")

# ───────────────────────── ④ 모델 평가 ─────────────────────────
with tabs[3]:
    st.subheader("5-fold 교차검증 RMSE 비교")
    st.write("RMSE는 예측값과 실제값의 차이를 cm로 나타낸 값으로, **작을수록** 관측 구간 안에서 잘 맞는다는 뜻이다. "
             "(KFold, n_splits=5, shuffle=True, random_state=0)")
    rows = [{"모델": c["name"], "남자 RMSE": c["cv_rmse"]["남자"], "여자 RMSE": c["cv_rmse"]["여자"],
             "남자 2040": c["pred_2040"]["남자"], "여자 2040": c["pred_2040"]["여자"],
             "출처": "노트북" if c["in_notebook"] else "웹앱 추가 계산"} for c in catalog]
    score_df = pd.DataFrame(rows)
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        for sex in ["남자", "여자"]:
            fig.add_bar(x=score_df["모델"], y=score_df[f"{sex} RMSE"], name=sex,
                        marker_color=SEX_COLOR[sex], text=score_df[f"{sex} RMSE"], textposition="outside")
        fig.update_layout(barmode="group", height=400, yaxis_title="CV RMSE (cm)", margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(score_df, hide_index=True, use_container_width=True)
    st.warning("**점수가 좋다고 미래도 잘 맞힐까?** 3차 다항회귀는 노트북의 세 모델 중 RMSE가 가장 낮지만, "
               "2040년에는 키가 오히려 줄어든다고 예측한다. 교차검증은 '데이터가 있는 범위 안'의 성적일 뿐 "
               "데이터 밖(미래) 예측력은 보장하지 않는다.")
    st.caption("※ 기준선·최근 직선·로지스틱 모델의 점수는 보고서 8.2 한계(“기준선과 추가 모델 점수가 없다”)를 보완하려고 "
               "웹앱에서 같은 방법으로 추가 계산했다. 최근 직선은 2010~2025년 자료로만 평가해 다른 모델과 직접 비교하기 어렵다.")

# ───────────────────────── ⑤ 2040 예측 체험 ─────────────────────────
with tabs[4]:
    st.subheader("모델을 골라 미래 평균 키를 예측해 보세요")
    ctrl, out = st.columns([1, 2])
    with ctrl:
        model_key = st.selectbox("모델 선택 (joblib 파일 교체)", list(name_by_key),
                                 index=list(name_by_key).index("poly1"),
                                 format_func=lambda k: name_by_key[k])
        target = st.slider("예측할 연도", 2026, 2060, 2040)
        compare = st.checkbox("다른 모델 곡선도 함께 보기", value=False)
        info = cat_by_key[model_key]
        st.caption(f"📂 `models/{model_key}.joblib` · 학습 구간 {info['train_from']}~2025")
        st.markdown(f"**모델 설명** — {info['desc']}")
        if model_key == "logistic":
            st.markdown(f"이 곡선이 수렴하는 상한: 남 **{info['ceiling']['남자']}cm**, 여 **{info['ceiling']['여자']}cm**")

    years_obs = nat.index.values
    years_line = np.arange(1975, 2061)
    with out:
        m1, m2 = st.columns(2)
        for col, sex in zip([m1, m2], ["남자", "여자"]):
            p = float(predict(model_key, sex, [target])[0])
            col.metric(f"{target}년 {sex} 예측", f"{p:.1f} cm",
                       f"{p - nat.loc[2025, sex]:+.1f} cm (2025년 대비)")
        fig = go.Figure()
        keys = list(name_by_key) if compare else [model_key]
        for sex in ["남자", "여자"]:
            fig.add_scatter(x=years_obs, y=nat[sex], mode="markers", name=f"{sex} 실제",
                            marker=dict(color=SEX_COLOR[sex], size=5))
            for k in keys:
                sel = k == model_key
                fig.add_scatter(x=years_line, y=predict(k, sex, years_line), mode="lines",
                                name=f"{sex} · {name_by_key[k]}", showlegend=sex == "남자" or not compare,
                                legendgroup=k,
                                line=dict(color=SEX_COLOR[sex] if sel else MODEL_COLOR[k],
                                          width=3 if sel else 1.3, dash=None if sel else "dot"))
            fig.add_scatter(x=[target], y=predict(model_key, sex, [target]), mode="markers+text",
                            marker=dict(color=SEX_COLOR[sex], size=13, symbol="star"),
                            text=[f"{predict(model_key, sex, [target])[0]:.1f}"],
                            textposition="top center", showlegend=False)
        fig.add_vrect(x0=2025.5, x1=2060, fillcolor="#e2e8f0", opacity=0.35, line_width=0,
                      annotation_text="데이터 밖 (외삽)", annotation_position="top left")
        fig.update_layout(height=480, yaxis_title="평균 키 (cm)", xaxis_title="연도",
                          margin=dict(t=20), legend=dict(orientation="h", y=-0.18))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**{target}년 모델별 예측 한눈에 보기**")
    tbl = pd.DataFrame([{"모델": name_by_key[k],
                         "남자 (cm)": round(float(predict(k, "남자", [target])[0]), 1),
                         "여자 (cm)": round(float(predict(k, "여자", [target])[0]), 1)} for k in name_by_key])
    st.dataframe(tbl, hide_index=True, use_container_width=True)

# ───────────────────────── ⑥ 결론·한계 ─────────────────────────
with tabs[5]:
    st.subheader("결론")
    st.success("1975~2025년 자료를 분석한 결과, 17세 고등학생의 평균 키는 과거에 빠르게 증가했으나 최근 20년은 거의 "
               "정체되었다. 팀은 **1차 다항회귀(직선) 모델**로 2040년 평균 키를 **남자 178.0cm, 여자 163.8cm**로 예상했다. "
               "다만 2040년 예측값은 모델에 따라 남자 170.3~178.0cm, 여자 158.9~163.8cm로 크게 달라, "
               "어느 하나를 확정적인 정답으로 보기 어렵다. 여러 모델의 결과를 함께 제시하고 불확실성을 고려해야 한다.")
    st.subheader("한계와 보완 방법")
    st.markdown("""
| 구분 | 한계 | 보완 방법 |
|---|---|---|
| 데이터 | 17개 시·도를 단순평균해 학생 수 차이를 반영하지 못함. 2020년 자료 없음 | KOSIS '시도별 = 계' 자료 사용, 추가 연도 확보 |
| 모델 | 연도만 특성으로 사용 — 영양·수면·유전 등 요인 미반영. 외삽이 불안정 | 관련 자료 추가, 성장 한계를 고려한 모델 검증 |
| 평가 | 시간 순서를 섞은 KFold라 미래 예측력 검증에 한계 | 과거로 학습 → 최근 연도로 평가하는 시간 순서 검증 적용 |
""")
    st.subheader("윤리 점검")
    st.markdown("✅ 개인 식별 정보 없음 · ✅ 특정 집단에 불리한 사용 검토 · ✅ 모델 판단 이유 설명 가능 · "
                "✅ 출처·이용 조건 준수 · ✅ 과장 없이 한계 함께 제시")
    st.subheader("다음에 해 보고 싶은 것")
    st.write("시도별 '계' 자료로 학생 수를 반영한 전국 평균을 계산하고, 시간 순서를 고려한 검증으로 기준선·로지스틱·"
             "최근 직선 모델을 같은 기준에서 비교해 예측의 신뢰도를 높이고 싶다.")
    st.caption("참고: 「학생건강검사 표본통계: 시·도별 학생표본 신체(키) 검사 현황」, 국가통계포털(KOSIS), 1975~2025년.")
