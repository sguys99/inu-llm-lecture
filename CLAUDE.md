# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude Code(claude.ai/code)를 위한 가이드입니다.
사용자용 상세 안내는 [`README.md`](README.md)를 먼저 참고하세요.

## 프로젝트 개요

인천대학교(INU) 비교과 과정 **"LLM을 활용한 어플리케이션 개발"** 교육 자료입니다. 산출물은 두 가지이며, 둘은 공용 소스 코드 `src/app/`을 함께 사용합니다.

- **`notebooks/`** — 모델 호출 → 프롬프트 엔지니어링 → RAG → LangGraph로 이어지는 단계별 학습 노트북
- **`demo/`** — 배운 내용을 묶은 Streamlit 멀티페이지 데모 앱
- **`src/app/`** — 노트북·데모가 공유하는 LLM/임베딩/RAG 유틸리티 (설치형 패키지)

## 개발 환경 / 자주 쓰는 명령

패키지 관리는 **`uv`** (pip 아님)를 사용하고, Python은 `3.12.12`로 핀 고정됩니다.

```bash
# 환경 구성
make init          # 운영 환경 (uv python pin → venv → uv sync)
make init-dev      # 개발 도구 포함 (ruff, pre-commit, pytest) + pre-commit 훅 설치
source .venv/bin/activate
make help          # 사용 가능한 make 타깃 목록

# 데모 앱 실행 (브라우저 http://localhost:8501, 로그인 STREAMLIT_ID/STREAMLIT_PW, 기본 admin/admin)
streamlit run demo/admin.py

# 노트북
jupyter lab

# 린트 / 포맷 (line-length 105, target py312)
uv run ruff check --fix src/
uv run ruff format src/

# 테스트 (주의: tests/ 디렉터리는 현재 삭제된 상태 — 필요 시 재작성)
uv run pytest
uv run pytest --cov=src/app
```

커밋 시 pre-commit 훅이 자동 실행됩니다(trailing whitespace, EOF, ruff `--fix`/format, add-trailing-comma 등). `notebooks/`, `data/`, `.venv/`, `.claude/`, `uv.lock`은 훅에서 제외됩니다. Python 코드를 추가/수정하면 커밋 전에 `uv run ruff check --fix`와 `format`을 돌려 두는 것이 좋습니다.

## 아키텍처 / 코드 구조

### 공용 패키지 `src/app/`

`src/app/`은 설치형 패키지입니다(`pyproject.toml`의 `[tool.hatch.build.targets.wheel] packages = ["src/app"]`). `uv sync` 후에는 데모·노트북 어디서든 `from app... import ...` 형태로 임포트합니다(ruff isort `known-first-party = ["app"]`). 새 코드를 작성할 때는 아래 기존 함수들을 먼저 재사용하세요.

| 모듈 | 주요 진입점 | 설명 |
|------|-------------|------|
| `app.modules.llms` | `get_llm()`, `get_embedding()` | 모델명 접두사로 provider 분기 (`claude-*`→Anthropic, `gpt-*`→OpenAI, `voyage-*`→Voyage 등). `RateLimitedVoyageEmbeddings`로 무료 등급 레이트리밋 자동 재시도. |
| `app.modules.vector_db` | `get_vector_store()`, `get_splitter()`, `build_faiss_throttled()` | FAISS/Chroma 생성, 텍스트 분할기 선택, 레이트리밋 대응 배치 임베딩. |
| `app.modules.prompts` | `build_qa_prompt()`, `save_prompt()`, `save_fewshot_prompt()` | QA/Few-shot 프롬프트 조립 및 YAML 직렬화. |
| `app.modules.documents` | `get_pdf_loader()`, `get_docx_loader()` | loader 종류(pymupdf/pypdf/pdfplumber/pdfminer, docx2txt) 선택. |
| `app.chains` | `build_simple_chain()`, `build_memory_chain()` | retriever→format→prompt→llm 체인 조립 패턴. |
| `app.utils.path` | `REPO_ROOT`, `CONFIG_PATH`, `DATA_PATH`, `LOG_PATH`, `PROMPT_CONFIG_PATH` 등 | 경로 상수. 경로를 하드코딩하지 말고 이 모듈을 사용. |
| `app.utils.config_loader` | `load_config()`, `load_yaml()`, `dump_yaml()` | YAML 설정 로딩. |
| `app.utils.rag_utils` | `format_docs()`, `format_docs_with_source()`, `save_rag_configs()` | 문서 포맷팅, RAG 설정 저장. |

### 데모 앱 `demo/`

- 진입점 [`demo/admin.py`](demo/admin.py) — `st.Page` 기반 멀티페이지 + 로그인 게이트.
- [`demo/1_single_doc_application/qa_application.py`](demo/1_single_doc_application/qa_application.py) — PDF/DOCX 한 개 업로드 후 스트리밍 QA.
- [`demo/2_qa_rag_assistant/rag_settings.py`](demo/2_qa_rag_assistant/rag_settings.py) — 다중 문서로 RAG 구성(벡터DB·프롬프트·청크) 저장.
- [`demo/2_qa_rag_assistant/rag_application.py`](demo/2_qa_rag_assistant/rag_application.py) — 저장한 구성으로 대화형 질의응답.
- 공용 헬퍼: [`demo/function_utils.py`](demo/function_utils.py)(`ChatCallbackHandler` 스트리밍, `embed_file()`, 문서 렌더링 등), [`demo/page_utils.py`](demo/page_utils.py)(`login()`/`logout()`).
- 데모 내부 임포트는 디렉터리 기준 상대 임포트(`from page_utils import login`)입니다 — Streamlit이 페이지 기준으로 실행하기 때문.

### 데이터 / 설정

- 프롬프트 템플릿: [`configs/prompt_template/`](configs/prompt_template) (`simple_qa_prompt.yaml`, `simple_qa_prompt_kor.yaml`, `example_template.yaml`). `langchain_core.prompts.load_prompt(path, encoding="utf-8")`로 로드.
- 실습 데이터: [`data/osh_act.pdf`](data/osh_act.pdf)(산업안전보건법) + 사전 빌드 인덱스 [`data/vector_db/osh_act_faiss/`](data/vector_db/osh_act_faiss).
- 저장된 RAG 구성/로그는 `logs/rag/`에 디렉터리 단위로 생성됩니다.

## 환경 변수

[`.env.example`](.env.example)을 `.env`로 복사해 채웁니다. 모든 진입점에서 `load_dotenv()`를 호출합니다.

```env
ANTHROPIC_API_KEY=                     # Anthropic Claude
ANTHROPIC_MODEL=claude-haiku-4-5       # 기본 LLM
VOYAGE_API_KEY=                        # Voyage AI
VOYAGE_EMBEDDING_MODEL=voyage-4-lite   # 기본 임베딩
STREAMLIT_ID=admin                     # 데모 로그인 계정
STREAMLIT_PW=admin
```

## 컨벤션 / 확장 포인트

- **언어**: README·주석·문서·코드 리뷰는 한국어 관례를 따릅니다. 식별자/명령어/코드는 영문.
- **기본 모델**: LLM `claude-haiku-4-5`, 임베딩 `voyage-4-lite`. 최신·고성능이 필요하면 Claude 상위 모델(Opus/Sonnet) 고려.
- **새 LLM/임베딩 추가** → `app.modules.llms`의 `get_llm()`/`get_embedding()`에 분기 추가.
- **새 프롬프트 추가** → `configs/prompt_template/`에 YAML 추가 후 `rag_settings.py`의 selectbox에 등록.
- **새 데모 페이지 추가** → `demo/`에 `.py` 추가 후 `admin.py` 네비게이션에 `st.Page`로 등록.

## 라이선스

[Apache License 2.0](LICENSE).
