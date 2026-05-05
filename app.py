
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Page setting
# -----------------------------
st.set_page_config(
    page_title="탐구 설계 점검 AI",
    page_icon="🔎",
    layout="wide"
)

# -----------------------------
# 2. Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #F8FAFF 0%, #FFFFFF 45%);
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }

    .hero-box {
        padding: 2rem 2.2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #EEF4FF 0%, #F7F2FF 55%, #FFFFFF 100%);
        border: 1px solid #E3E8F7;
        box-shadow: 0 10px 28px rgba(62, 91, 180, 0.08);
        margin-bottom: 1.4rem;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: #172554;
        margin-bottom: 0.45rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.7;
        color: #475569;
        margin-bottom: 0;
    }

    .section-card {
        padding: 1.3rem 1.4rem;
        border-radius: 20px;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        margin: 1rem 0;
    }

    .mini-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        min-height: 105px;
    }

    .mini-title {
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.35rem;
    }

    .mini-text {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    .badge {
        display: inline-block;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        background: #DBEAFE;
        color: #1D4ED8;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .warning-note {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        color: #92400E;
        line-height: 1.6;
        margin-top: 0.7rem;
    }

    .good-note {
        padding: 1rem 1.2rem;
        border-radius: 16px;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        line-height: 1.6;
        margin-top: 0.7rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 1rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }

    div.stButton > button {
        border-radius: 14px;
        padding: 0.75rem 1.1rem;
        font-weight: 700;
        border: 0;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
    }

    textarea, input {
        border-radius: 14px !important;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.9rem;
        padding-top: 1.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 3. Hero
# -----------------------------
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">탐구 설계 점검 AI</div>
        <p class="hero-subtitle">
        탐구 주제를 입력하면 머신러닝 모델이 탐구 활동 유형을 추천하고,
        탐구 대상·원인·결과·분석 기준·한계 요소를 점검해 주는 학습 보조 웹앱입니다.
        </p>
        <div style="margin-top:1rem;">
            <span class="badge">Machine Learning</span>
            <span class="badge">Research Design</span>
            <span class="badge">Streamlit Cloud</span>
            <span class="badge">No Report Ghostwriting</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 4. Training data
# -----------------------------
train_topics = [
    # 설문조사
    "AI 사용 빈도가 고등학생의 학습 태도에 미치는 영향",
    "청소년의 스마트폰 사용 시간이 집중력에 미치는 영향",
    "학생들의 생성형 AI 사용에 대한 인식 조사",
    "학교 급식 만족도와 메뉴 선호도 조사",
    "자율주행자동차에 대한 학생들의 신뢰도 조사",
    "고등학생의 챗봇 사용 경험과 학습 만족도의 관계",
    "학생들의 디지털 감시 사회에 대한 찬반 인식 조사",
    "AI 추천 시스템에 대한 소비자의 신뢰도 조사",
    "고등학생의 진로 선택 기준에 대한 인식 조사",
    "청소년의 SNS 사용 목적과 정서 변화에 대한 설문 조사",
    "학생들의 온라인 수업 만족도와 학습 참여도 조사",
    "학교 도서관 이용 경험에 대한 학생 인식 조사",

    # 실험
    "빛의 색에 따른 검정말의 광합성량 변화",
    "온도 변화가 효소 반응 속도에 미치는 영향",
    "소음 환경이 기억력 테스트 결과에 미치는 영향",
    "물의 양에 따른 식물 성장 비교",
    "운동 전후 심박수 변화 측정",
    "학습 시간 차이에 따른 단어 암기 결과 비교",
    "조명 밝기에 따른 집중력 테스트 결과 변화",
    "휴식 시간 유무에 따른 문제 해결 시간 비교",
    "스마트폰 사용 전후 집중력 변화 실험",
    "음악 청취 여부에 따른 계산 문제 풀이 속도 비교",
    "수면 시간 차이에 따른 기억력 테스트 결과 비교",
    "카페인 섭취 전후 반응 속도 변화 측정",

    # 사례 비교
    "AI 챗봇 사고 사례와 책임 소재 비교",
    "스웨덴 난민 정책의 변화 사례 비교",
    "자율주행차 사고 사례와 안전 대책 비교",
    "생성형 AI 환각 사례와 해결 방안 비교",
    "디지털 감시 사회의 찬반 근거 비교",
    "국가별 AI 규제 정책 사례 비교",
    "로봇 돌봄 서비스의 장점과 한계 사례 비교",
    "AI 편향으로 인한 차별 사례 비교",
    "국내외 개인정보 보호 정책 사례 비교",
    "학교 스마트폰 사용 제한 정책 사례 비교",
    "친환경 에너지 정책의 국가별 차이 비교",
    "플랫폼 기업의 추천 알고리즘 문제 사례 비교",

    # 데이터 분석
    "유튜브 댓글을 활용한 AI 인식 데이터 분석",
    "설문 응답 데이터를 활용한 학생 의견 분석",
    "공공데이터를 활용한 지역별 환경 문제 분석",
    "댓글 데이터에서 자주 등장하는 쟁점 분석",
    "뉴스 제목 데이터를 활용한 사회 이슈 변화 분석",
    "학교 설문 결과를 활용한 스마트폰 사용 실태 분석",
    "기후변화 통계 자료를 활용한 연도별 변화 분석",
    "리뷰 데이터를 활용한 제품 만족도 분석",
    "공공데이터를 활용한 교통사고 발생 지역 분석",
    "기상 데이터를 활용한 기온 변화 시각화",
    "도서 대출 데이터를 활용한 학생 독서 경향 분석",
    "학급 설문 데이터를 활용한 학습 습관 분석",

    # 문헌조사
    "AI 윤리에 관한 선행 연구 문헌 조사",
    "뇌의 보상회로와 강화학습의 관련성 문헌 조사",
    "인공지능 신뢰성에 관한 이론 조사",
    "기후변화 원인에 관한 자료 조사",
    "로봇 기술 발전과 인간 노동의 관계 조사",
    "생성형 AI의 환각 현상 원인에 관한 자료 조사",
    "얼굴 인식 기술의 기하학적 원리에 관한 문헌 조사",
    "자연어 처리 기술의 발전 과정 조사",
    "자율주행자동차의 센서 원리에 관한 자료 조사",
    "인공지능 추천 시스템의 작동 원리 조사",
    "AI와 저작권 문제에 관한 문헌 조사",
    "디지털 격차 문제에 관한 선행 연구 조사",

    # 프로그램 개발
    "탐구 주제 구체화를 돕는 웹앱 개발",
    "코딩을 모르는 학생을 위한 탐구 설계 도우미 프로그램 제작",
    "학생들이 탐구 주제를 선택할 수 있도록 돕는 추천 시스템 개발",
    "사용자 입력을 바탕으로 탐구 방법을 추천하는 프로그램 구현",
    "Streamlit을 활용한 탐구 설계 점검 웹앱 제작",
    "GitHub와 Streamlit Cloud를 활용한 교육용 웹서비스 배포",
    "AI 기반 학습 보조 웹앱 개발",
    "사용자 맞춤형 탐구 활동 지원 프로그램 제작",
    "고등학생의 탐구 활동을 돕는 프로그램 개발",
    "탐구 보고서 작성 전 설계 요소를 점검하는 웹서비스 제작",
    "탐구 주제와 방법을 추천하는 머신러닝 프로그램 개발",
    "학교 수행평가 준비를 돕는 웹 기반 학습 도구 제작",
    "코랩 프로그램을 웹앱으로 변환하여 배포하기",
    "탐구 활동 주제 선택과 구체화를 지원하는 웹앱 만들기"
]

train_methods = (
    ["설문조사"] * 12
    + ["실험"] * 12
    + ["사례 비교"] * 12
    + ["데이터 분석"] * 12
    + ["문헌조사"] * 12
    + ["프로그램 개발"] * 14
)

# -----------------------------
# 5. ML model
# -----------------------------
@st.cache_resource
def train_model():
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB())
    ])
    model.fit(train_topics, train_methods)
    return model

model = train_model()

# -----------------------------
# 6. Functions
# -----------------------------
def get_reason(predicted_method):
    reasons = {
        "설문조사": "사람의 인식, 태도, 만족도, 신뢰도 등을 확인하는 데 적합하므로 설문조사를 추천합니다.",
        "실험": "조건을 바꾸고 결과 변화를 비교할 수 있으므로 실험 방법을 추천합니다.",
        "사례 비교": "여러 사례의 공통점과 차이점을 비교하는 데 적합하므로 사례 비교를 추천합니다.",
        "데이터 분석": "댓글, 설문 응답, 공공데이터 등 자료를 수집해 수치화할 수 있으므로 데이터 분석을 추천합니다.",
        "문헌조사": "개념, 이론, 선행 연구를 정리하는 데 적합하므로 문헌조사를 추천합니다.",
        "프로그램 개발": "사용자의 문제를 해결하는 프로그램이나 웹서비스 제작에 초점이 있으므로 프로그램 개발형 탐구를 추천합니다."
    }
    return reasons.get(predicted_method, "입력 내용을 바탕으로 가장 가까운 탐구 활동 유형을 추천했습니다.")

def design_check(topic, idea):
    text = topic + " " + idea
    score = 0
    feedback = []
    questions = []

    checks = [
        (
            "탐구 대상",
            ["학생", "고등학생", "청소년", "교사", "사용자", "소비자", "대중", "노인", "아동", "친구", "학습자", "사람"],
            "탐구 대상이 드러나 있어 조사 범위나 사용자 범위를 설정하기 쉽습니다.",
            "탐구 대상이 명확하지 않습니다.",
            "누구를 대상으로 탐구하거나 사용할 프로그램인가요?"
        ),
        (
            "원인·조건 요소",
            ["사용", "빈도", "시간", "방식", "노출", "활용", "경험", "의존", "학습", "기술", "환경", "조건", "AI", "인공지능", "프로그램", "웹앱", "서비스", "도구", "시스템"],
            "원인, 조건, 사용 기술 또는 프로그램 요소가 일부 드러납니다.",
            "원인이나 조건에 해당하는 요소가 부족합니다.",
            "무엇이 결과에 영향을 준다고 보고 있나요? 또는 어떤 기능이 핵심인가요?"
        ),
        (
            "결과 요소",
            ["영향", "변화", "차이", "태도", "인식", "만족도", "효율", "정확도", "사고력", "집중력", "불안", "신뢰", "결과", "도움", "개선", "구체화", "추천", "배포", "접근성"],
            "탐구를 통해 확인하거나 개선하려는 결과 요소가 드러납니다.",
            "결과로 무엇을 확인하거나 개선할지 부족합니다.",
            "탐구를 통해 어떤 변화나 개선을 확인하고 싶은가요?"
        ),
        (
            "관계 구조",
            ["미치는 영향", "관계", "비교", "차이", "상관", "원인", "결과", "증가", "감소", "따라", "전후", "돕는", "지원", "해결"],
            "두 요소 사이의 관계 또는 문제 해결 구조가 보입니다.",
            "탐구 요소 간 관계가 명확하지 않습니다.",
            "A와 B의 관계를 볼 것인지, 문제와 해결 도구의 관계를 볼 것인지 정해야 합니다."
        ),
        (
            "분석·평가 기준",
            ["빈도", "비율", "평균", "점수", "응답", "그래프", "표", "정확도", "오차", "전후", "집단", "분류", "상관", "통계", "추천", "점검", "기준", "사용성", "평가"],
            "결과를 분석하거나 평가할 기준이 일부 포함되어 있습니다.",
            "분석 또는 평가 기준이 부족합니다.",
            "결과를 빈도, 비율, 평균, 사용자 평가, 정확도, 기능 작동 여부 중 어떤 방식으로 분석할 건가요?"
        ),
        (
            "한계·윤리 고려",
            ["한계", "오차", "편향", "표본", "응답자", "정확", "신뢰도", "개인차", "제한", "윤리", "개인정보", "오류", "비용", "접근성"],
            "탐구의 한계나 윤리적 고려를 인식하고 있습니다.",
            "탐구의 한계나 윤리적 고려가 아직 부족합니다.",
            "표본 수, 응답 편향, 개인정보, 자료 신뢰성, 모델 오류 중 어떤 한계가 있을 수 있나요?"
        ),
    ]

    for name, words, good, bad, question in checks:
        if any(word in text for word in words):
            score += 2
            feedback.append((name, "충족", good))
        else:
            feedback.append((name, "보완 필요", bad))
            questions.append(question)

    if score >= 10:
        level = "우수"
    elif score >= 6:
        level = "보통"
    else:
        level = "보완 필요"

    return score, level, feedback, questions

def make_next_steps(predicted_method):
    common = [
        "탐구 대상 또는 사용자를 구체적으로 정한다.",
        "해결하려는 문제 상황을 한 문장으로 정리한다.",
        "결과를 평가할 기준을 수치화하거나 체크리스트화한다.",
        "탐구의 한계와 윤리적 고려 사항을 정리한다."
    ]

    specific = {
        "설문조사": [
            "설문 대상, 문항 수, 응답 방식, 분석 기준을 정한다.",
            "응답 결과를 비율, 평균, 집단 차이 등으로 비교한다."
        ],
        "실험": [
            "독립변인, 종속변인, 통제변인을 구분한다.",
            "실험 전후 결과를 비교할 측정 기준을 정한다."
        ],
        "사례 비교": [
            "비교할 사례를 2개 이상 선정한다.",
            "공통 기준으로 장점, 한계, 해결 방안을 비교한다."
        ],
        "데이터 분석": [
            "사용할 데이터의 출처와 수집 방법을 정한다.",
            "빈도, 비율, 평균, 그래프 등 분석 방법을 정한다."
        ],
        "문헌조사": [
            "신뢰할 수 있는 자료의 출처를 정한다.",
            "자료를 단순 요약하지 않고 비교 기준을 세워 분석한다."
        ],
        "프로그램 개발": [
            "사용자 입력, 처리 과정, 출력 결과를 흐름도로 정리한다.",
            "기능이 실제로 사용자의 문제 해결에 도움이 되는지 평가 기준을 만든다.",
            "코랩 프로그램을 Streamlit 웹앱으로 변환하고 GitHub와 Streamlit Cloud로 배포한다."
        ]
    }

    return specific.get(predicted_method, []) + common

# -----------------------------
# 7. Sidebar
# -----------------------------
with st.sidebar:
    st.header("앱 소개")
    st.write("보고서를 대신 작성하지 않고, 탐구 설계를 점검하는 학습 보조 도구입니다.")
    st.divider()
    st.subheader("사용 기술")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- scikit-learn")
    st.write("- CountVectorizer")
    st.write("- Naive Bayes")
    st.divider()
    st.subheader("추천 유형")
    st.write("설문조사, 실험, 사례 비교, 데이터 분석, 문헌조사, 프로그램 개발")

# -----------------------------
# 8. Intro cards
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("이 웹앱이 하는 일")

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
        <div class="mini-card">
            <div class="mini-title">1. 유형 추천</div>
            <div class="mini-text">입력한 탐구 주제를 머신러닝 모델로 분류하여 적절한 탐구 활동 유형을 추천합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_b:
    st.markdown(
        """
        <div class="mini-card">
            <div class="mini-title">2. 설계 점검</div>
            <div class="mini-text">대상, 원인, 결과, 관계, 분석 기준, 한계 요소가 포함되어 있는지 확인합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_c:
    st.markdown(
        """
        <div class="mini-card">
            <div class="mini-title">3. 보완 질문</div>
            <div class="mini-text">부족한 항목에 대해 사용자가 직접 생각해 볼 수 있는 질문을 제공합니다.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 9. User input
# -----------------------------
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.subheader("1. 탐구 주제 입력")

topic = st.text_input(
    "탐구 주제를 입력하세요",
    placeholder="예: 탐구 설계 웹앱 만들기"
)

idea = st.text_area(
    "현재 생각하고 있는 탐구 내용을 적어주세요",
    placeholder="예: 고등학생들이 탐구 활동 주제를 선택하고 구체화하는 것을 돕는 프로그램을 만들어 배포하기",
    height=140
)

run = st.button("탐구 설계 점검하기", type="primary")
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 10. Output
# -----------------------------
if run:
    if not topic.strip() or not idea.strip():
        st.warning("탐구 주제와 현재 생각을 모두 입력해 주세요.")
    else:
        predicted_method = model.predict([topic + " " + idea])[0]
        probabilities = model.predict_proba([topic + " " + idea])[0]
        classes = model.classes_

        score, level, feedback, questions = design_check(topic, idea)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("2. 머신러닝 기반 탐구 활동 유형 추천 결과")

        col1, col2, col3 = st.columns(3)
        col1.metric("추천 유형", predicted_method)
        col2.metric("탐구 설계 점수", f"{score} / 12점")
        col3.metric("설계 수준", level)

        st.markdown(f'<div class="good-note"><b>추천 이유</b><br>{get_reason(predicted_method)}</div>', unsafe_allow_html=True)

        prob_df = pd.DataFrame({
            "탐구 활동 유형": classes,
            "가능성": probabilities
        }).sort_values("가능성", ascending=False)

        st.write("탐구 활동 유형별 가능성")
        st.dataframe(prob_df, use_container_width=True, hide_index=True)
        st.bar_chart(prob_df.set_index("탐구 활동 유형"))

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("3. 탐구 설계 요소 점검 결과")

        feedback_df = pd.DataFrame(feedback, columns=["점검 항목", "상태", "점검 결과"])
        st.dataframe(feedback_df, use_container_width=True, hide_index=True)

        st.subheader("4. 보완 질문")
        if questions:
            for q in questions:
                st.write("- " + q)
        else:
            st.success("현재 탐구 설계가 비교적 잘 구성되어 있습니다. 실제 자료 수집 계획이나 사용자 평가 계획을 더 구체화해 보세요.")

        st.subheader("5. 다음 단계")
        for step in make_next_steps(predicted_method):
            st.write("- " + step)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("6. 탐구 설계표 요약")

        summary_df = pd.DataFrame([
            ["탐구 주제", topic],
            ["현재 아이디어", idea],
            ["추천 탐구 활동 유형", predicted_method],
            ["탐구 설계 수준", level],
            ["탐구 설계 점수", f"{score} / 12점"]
        ], columns=["항목", "내용"])

        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        result_text = f"""탐구 설계 점검 결과

탐구 주제: {topic}
현재 아이디어: {idea}
추천 탐구 활동 유형: {predicted_method}
탐구 설계 점수: {score} / 12점
탐구 설계 수준: {level}

[세부 점검 의견]
""" + "\n".join([f"- {item[0]}({item[1]}): {item[2]}" for item in feedback]) + """

[보완 질문]
""" + ("\n".join([f"- {q}" for q in questions]) if questions else "- 현재 탐구 설계가 비교적 잘 구성되어 있습니다.") + """

[다음 단계]
""" + "\n".join([f"- {s}" for s in make_next_steps(predicted_method)])

        st.download_button(
            label="결과 TXT 다운로드",
            data=result_text,
            file_name="research_design_check_result.txt",
            mime="text/plain"
        )

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="warning-note">탐구 주제와 현재 생각을 입력한 뒤, <b>탐구 설계 점검하기</b> 버튼을 눌러 주세요.</div>',
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="footer">
    이 앱은 생성형 AI가 보고서를 대신 작성하는 도구가 아니라, 탐구 설계 요소를 점검하고 보완 질문을 제공하는 학습 보조 도구입니다.
    </div>
    """,
    unsafe_allow_html=True
)
