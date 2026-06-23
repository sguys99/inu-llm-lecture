# Demo/Src 코드 적응 계획 — claude-haiku-4-5 + voyage-4-lite + FAISS

## 배경
`demo/`·`src/app/`는 다른 프로젝트(Ollama/OpenAI 기반)에서 복사해온 Streamlit 대시보드다.
본 강의 프로젝트는 노트북(2.1~3.1)과 동일하게 **Claude Haiku 4.5(LLM) + Voyage 4-lite(임베딩) + FAISS** 스택을 쓴다.
`streamlit run admin.py`로 실행되는 대시보드가 위 스택으로 동작하고, 임베딩은 노트북 2.1 패턴
(20페이지 제한 + 레이트리밋 배치)으로 안전하게 생성되도록 적응한다.

## 확정 사항
- 실행 진입점: `demo/admin.py` (MAIN.py·pages/01,02는 레거시 → 제외)
- 수정 범위: 단일문서 LLM 앱(simple_llm, simple_llm_law, qa_application) + RAG QA assistant(rag_settings, rag_application)
- 제거: 데이터분석 에이전트 전부 → `3_data_analytics_agent/*` + `da_application`(Simple DA Agent)
- 모델 UI: 드롭다운 유지(단일 항목 claude-haiku-4-5 / voyage-4-lite)
- 레이트리밋: 노트북 2.1 패턴 완전 이식(RateLimitedVoyageEmbeddings + 20페이지 + BATCH=6/SLEEP=60)
- 벡터 인덱스: 사용자 업로드 문서 기반

---

## 작업 단계

### Phase 0 — 진행 추적 / 환경 확인
- [x] 저장소 루트에 `demo-plan.md` 생성
- [x] `.env`에 ANTHROPIC_API_KEY / ANTHROPIC_MODEL / VOYAGE_API_KEY / VOYAGE_EMBEDDING_MODEL 존재 확인

### Phase 1 — 핵심 모듈
**src/app/modules/llms.py**
- [x] `VoyageAIEmbeddings`, `time` import 추가
- [x] `RateLimitedVoyageEmbeddings` 클래스 이식(노트북 2.1)
- [x] `get_embedding()`에 voyage 분기 + 기본값 `voyage-4-lite`
- [x] `get_llm()` 기본값 `gpt-4o` → `claude-haiku-4-5`
- [x] Ollama import 지연화(미설치 패키지로 인한 import 오류 방지)

**src/app/modules/vector_db.py**
- [x] `build_faiss_throttled()` 배치+throttle FAISS 빌더 추가(BATCH=6/SLEEP=60)

### Phase 2 — demo/function_utils.py
- [x] `embed_file()` / `embed_file_with_cache()`: 기본 임베딩 voyage-4-lite, base_url 제거, 20페이지 제한, throttled 빌더 사용
- [x] `load_retriver()`: 기본 임베딩 voyage-4-lite, base_url 제거

### Phase 3 — RAG QA assistant
**rag_settings.py**
- [x] 임베딩 voyage-4-lite, 20페이지 제한, throttled 빌더 + status 진행률
- [x] config `embedding_type` → voyage-4-lite
- [x] 임베딩 소요시간 안내

**rag_application.py**
- [x] LLM 드롭다운 → `["claude-haiku-4-5"]`, 선택 로직 단순화
- [x] get_llm/load_retriver의 base_url 제거

### Phase 4 — 단일문서 LLM 앱
- [x] simple_llm.py: 드롭다운 단일항목, 선택 로직 단순화, base_url/num_predict 제거
- [x] simple_llm_law.py: 동일
- [x] qa_application.py: 드롭다운 단일항목, 임베딩 voyage-4-lite, base_url 제거

### Phase 5 — admin.py 데이터분석 에이전트 제거
- [x] `da_application` 페이지 정의 제거
- [x] `qa_agent_settings`·`qa_agent_application` 페이지 정의 제거
- [x] 네비게이션에서 da_application 제거 + "3️⃣ 데이터분석 에이전트" 줄 삭제

### Phase 6 — 검증
- [x] 전체 수정 파일 `py_compile` 통과
- [x] `get_llm('claude-haiku-4-5')` → ChatAnthropic, `get_embedding('voyage-4-lite')` → RateLimitedVoyageEmbeddings 생성 확인
- [x] `streamlit run admin.py` 헤드리스 부팅 성공(에러 없음), 네비게이션에서 데이터분석 항목 제거 확인
- [x] `function_utils` import 배선(build_faiss_throttled/embed_file) 정상
- [ ] (수동, 실제 키 필요) Simple LLM / LLM LAW: claude 스트리밍 응답
- [ ] (수동) RAG Settings: PDF 업로드 → 20p 제한·배치 임베딩 진행 → FAISS 인덱스 생성
- [ ] (수동) RAG Application: 저장 RAG 선택 → 검색+claude 답변
- [ ] (수동) Simple QA Application: PDF 업로드 → voyage 임베딩(20p) → 질문/답변

### 패키지 리네임 (lfc_rag → app)
- `src/lfc_rag/` → `src/app/` 로 디렉터리 이름 변경, 모든 `from lfc_rag...` → `from app...` 치환
- `pyproject.toml`: `[tool.hatch.build.targets.wheel] packages = ["src/app"]`, `[tool.coverage.run] source = ["src/app"]`
- `uv sync` 로 editable 재설치 → `.pth`가 `src`를 가리켜 **PYTHONPATH 없이** `import app` 동작
- (참고) `demo/agent_utils.py`는 제거된 데이터분석 에이전트 전용이며 `matplotlib` 미설치라 import 불가 — 활성 앱과 무관

### 실행 방법 / 주의 (중요)
- 데모 실행(PYTHONPATH 불필요):
  ```bash
  cd demo
  uv run streamlit run admin.py
  ```
- `.env`의 `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`에 **실제 키**를 채워야 LLM/임베딩이 동작
- Voyage 무료 한도로 임베딩 생성은 문서당 약 4~5분 소요(배치+대기)
