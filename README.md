# INU LLM Application Lecture

인천대학교(INU) 비교과 과정 **"LLM을 활용한 어플리케이션 개발"** 의 교육 자료입니다. LLM을 처음 다뤄 보는 수강생도 모델 호출에서 시작해 프롬프트 엔지니어링, RAG, 에이전트까지 단계적으로 익히고, 마지막에는 직접 동작하는 LLM 애플리케이션을 만들어 봅니다.

자료는 크게 두 가지로 구성됩니다.

- **`notebooks/`** — 개념을 하나씩 실습하는 단계별 학습 노트북
- **`demo/`** — 배운 내용을 실제 서비스 형태로 묶은 Streamlit 데모 애플리케이션

---

## 무엇을 배우나

노트북은 기초, RAG, 워크플로우 순서로 이어집니다. 위에서부터 차례대로 따라가면 됩니다.

### 1장. LLM 기초와 프롬프트 엔지니어링

| 노트북 | 다루는 내용 |
|--------|-------------|
| [`1.1_foundation_models.ipynb`](notebooks/1.1_foundation_models.ipynb) | Claude 모델 호출, 메시지 구조(System/Human/AI), 에이전트 개요 |
| [`1.2_prompt_engineering.ipynb`](notebooks/1.2_prompt_engineering.ipynb) | 시스템 프롬프트, Few-shot, Structured Output — 식당 리뷰 감성 분석 실습 |
| [`1.3_prompt_engineering(advanced).ipynb`](notebooks/1.3_prompt_engineering%28advanced%29.ipynb) | Chain of Thought(CoT), ReAct, `@tool` 도구 정의 |

### 2장. RAG와 벡터 DB

| 노트북 | 다루는 내용 |
|--------|-------------|
| [`2.1_vectordb.ipynb`](notebooks/2.1_vectordb.ipynb) | PDF 로드 → 텍스트 분할 → 임베딩 → FAISS 저장 및 유사도 검색 |
| [`2.2_rag.ipynb`](notebooks/2.2_rag.ipynb) | RAG 파이프라인 구성, 그라운딩(환각 방지), 출처 제시 |
| [`2.3_agentic_rag.ipynb`](notebooks/2.3_agentic_rag.ipynb) | 검색을 도구로 만든 동적 RAG, 질문에 따른 도구 라우팅 |

### 3장. LangGraph

| 노트북 | 다루는 내용 |
|--------|-------------|
| [`3.1_langgraph_rag.ipynb`](notebooks/3.1_langgraph_rag.ipynb) | LangGraph로 만드는 Agentic RAG — 검색·생성 흐름을 그래프로 정의 |

> 실습에는 [`data/osh_act.pdf`](data/osh_act.pdf)(산업안전보건법) 문서와, 미리 만들어 둔 FAISS 인덱스 [`data/vector_db/osh_act_faiss/`](data/vector_db/osh_act_faiss)를 사용합니다.

---

## 데모 애플리케이션

`demo/`는 노트북에서 배운 RAG와 프롬프트 기법을 Streamlit 멀티페이지 앱으로 묶은 것입니다. 진입점은 [`demo/admin.py`](demo/admin.py)이며, 로그인하면 아래 페이지로 이동합니다.

- **단일 문서 QA** ([`1_single_doc_application/`](demo/1_single_doc_application)) — PDF/DOCX 파일 하나를 업로드하면 문서 내용을 바탕으로 답변을 실시간 스트리밍합니다. Temperature, Chunk Size 등은 화면에서 조정합니다.
- **RAG 어시스턴트** ([`2_qa_rag_assistant/`](demo/2_qa_rag_assistant)) — 여러 문서를 한 번에 올려 벡터 DB를 구성(`rag_settings.py`)하고, 저장한 구성으로 대화형 질의응답(`rag_application.py`)을 합니다. 시스템 프롬프트와 Few-shot 예시, PDF 로더, 텍스트 분할기까지 직접 고릅니다.

---

## 기술 스택

| 구분 | 사용 기술 |
|------|-----------|
| LLM | Anthropic Claude (`claude-haiku-4-5`) |
| 임베딩 | Voyage AI (`voyage-4-lite`) |
| 프레임워크 | LangChain · LangGraph |
| 벡터 DB | FAISS (로컬) |
| 데모 UI | Streamlit |
| 문서 처리 | PyMuPDF, PyPDF, Mammoth(DOCX) |
| 패키지 관리 | uv |

---

## 시작하기

### 요구 사항

- Python 3.12.12
- [uv](https://docs.astral.sh/uv/) (패키지 관리자)
- Anthropic API 키, Voyage AI API 키

### 1. 환경 구성

```bash
# 의존성 설치 및 가상환경 생성
make init          # 기본 환경
# 또는
make init-dev      # 개발 도구(ruff, pre-commit, pytest) 포함

# 가상환경 활성화
source .venv/bin/activate
```

### 2. 환경 변수 설정

[`.env.example`](.env.example)을 복사해 `.env`를 만들고 API 키를 채웁니다.

```bash
cp .env.example .env
```

```env
# Anthropic Claude (LLM)
ANTHROPIC_API_KEY=         # https://console.anthropic.com 에서 발급
ANTHROPIC_MODEL=claude-haiku-4-5

# Voyage AI (Embedding)
VOYAGE_API_KEY=            # https://www.voyageai.com 에서 발급
VOYAGE_EMBEDDING_MODEL=voyage-4-lite

# 데모 로그인 계정
STREAMLIT_ID=admin
STREAMLIT_PW=admin
```

---

## 실행 방법

### 노트북

Jupyter Lab이나 VS Code에서 `notebooks/` 폴더의 `.ipynb` 파일을 열고 셀을 순서대로 실행합니다.

```bash
jupyter lab
```

### 데모 앱

```bash
streamlit run demo/admin.py
```

브라우저에서 [http://localhost:8501](http://localhost:8501)로 접속한 뒤, `.env`에 설정한 계정(기본값 `admin` / `admin`)으로 로그인합니다.

---

## 프로젝트 구조

```
inu-llm-lecture/
├── notebooks/      # 단계별 학습 노트북 (1장~3장)
├── demo/           # Streamlit 데모 애플리케이션
├── data/           # 실습용 문서 및 FAISS 벡터 DB
├── configs/        # 프롬프트 등 설정 파일
├── src/            # 공용 소스 코드
├── Makefile        # init / init-dev 명령
└── .env.example    # 환경 변수 템플릿
```

---

## 라이선스

이 프로젝트는 [Apache License 2.0](LICENSE)을 따릅니다.
