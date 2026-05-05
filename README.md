# 탐구 설계 점검 AI

고등학생이 탐구 주제를 입력하면 머신러닝 모델이 탐구 활동 유형을 추천하고, 탐구 설계 요소를 점검하는 Streamlit 웹앱입니다.

## 기능

- 탐구 주제 입력
- 머신러닝 기반 탐구 활동 유형 추천
- 탐구 대상, 원인·조건, 결과, 관계 구조, 분석 기준, 한계 요소 점검
- 보완 질문 제시
- 다음 단계 안내
- 결과 TXT 다운로드
- 카드형 UI와 Streamlit 테마 적용

## 사용 기술

- Python
- Streamlit
- scikit-learn
- CountVectorizer
- Multinomial Naive Bayes

## 배포 방법

1. GitHub에 `app.py`, `requirements.txt`, `README.md`, `.streamlit/config.toml`을 업로드합니다.
2. Streamlit Community Cloud에서 새 앱을 생성합니다.
3. Repository, branch, main file path를 선택합니다.
4. main file path는 `app.py`로 설정합니다.
5. Deploy 버튼을 누릅니다.


## 테마 파일 안내

이 버전은 숨김 폴더 없이 업로드하기 쉽게 구성했습니다.

- `app.py` 안에 카드형 UI와 색상 CSS가 이미 들어 있어서 이 파일만으로도 테마가 적용됩니다.
- `streamlit_theme_config.toml`은 참고용입니다.
- Streamlit 기본 테마까지 적용하고 싶다면 GitHub에서 `.streamlit` 폴더를 새로 만들고, 그 안에 `config.toml` 파일을 만든 뒤 `streamlit_theme_config.toml` 내용을 복사하면 됩니다.
