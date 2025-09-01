# AI 코딩 어시스턴트 프로젝트

**고성능 AI 코딩 어시스턴트 RAG(검색 증강 생성) 시스템**

이 프로젝트는 Triton Inference Server를 통해 고성능 LLM을 서빙하고, FastAPI와 LangChain을 기반으로 구축된 지능형 RAG 파이프라인을 통해 코드베이스에 대한 질문에 답변하는 AI 코딩 어시스턴트입니다.

## ✨ 주요 기능

* **고성능 LLM 서빙**: NVIDIA Triton Inference Server와 vLLM/TensorRT-LLM 백엔드를 사용하여 대규모 언어 모델을 효율적으로 서빙합니다.
* **지능형 코드 분석**: `tree-sitter`를 활용한 AST(추상 구문 트리) 기반 코드 분할로 코드의 의미론적 단위를 정확하게 파악합니다.
* **계층적 RAG (RAPTOR)**: Naive RAG를 넘어, 코드의 세부 내용과 전체적인 맥락을 모두 이해하는 RAPTOR 아키텍처를 적용하여 검색 정확도를 극대화합니다.
* **모듈식 파이프라인**: 검색, 프롬프트 구성, 추론 요청 등 각 단계가 명확히 분리된 파이프라인으로 유지보수와 확장이 용이합니다.
* **유연한 검색 전략**: `Naive`, `RAPTOR (숲과 나무)`, `RAPTOR (스코프 검색)` 등 다양한 RAG 검색 전략을 API 요청 시 선택적으로 사용할 수 있습니다.

## 📂 프로젝트 디렉토리 구조

```
SLproject/
├── server/
│   ├── run.sh                  # Triton 추론 서버 실행 스크립트
│   └── models/                 # Triton 모델 리포지토리
│       └── Nxcode/
│           ├── config.pbtxt
│           └── 1/
│               └── model.json
│
└── client/
├── app/                    # FastAPI 웹 애플리케이션
│   ├── main.py
│   └── router/
│       └── chat.py
├── core/                   # 핵심 비즈니스 로직 (파이프라인, RAG 컴포넌트)
│   └── ...
├── scripts/
│   └── ingest.py           # RAG 데이터베이스 생성 스크립트
├── vector_stores/          # 생성된 벡터 DB 저장 위치
├── test/                   # 인덱싱 테스트용 샘플 코드
└── requirements.txt        # Python 의존성 패키지
```

## 🚀 설치 및 실행 가이드

이 시스템은 **1) 추론 서버**와 **2) 클라이언트 서버** 두 부분으로 나뉘어 실행됩니다.

 ### 1. 추론 서버 (Triton) 구동 방법

추론 서버는 Docker 컨테이너로 실행됩니다.

1.  **모델 리포지토리 구성**:
    `server/models/` 디렉토리 아래에 사용할 LLM 모델의 설정을 구성합니다. (`config.pbtxt`, `model.json`)

2.  **실행 스크립트 권한 부여**:
    터미널을 열고 `server` 디렉토리로 이동하여 `run.sh` 스크립트에 실행 권한을 부여합니다. (최초 한 번만)
    ```bash
    cd server/
    chmod +x run.sh
    ```

3.  **서버 실행**:
    스크립트를 실행하여 Triton 서버 컨테이너를 시작합니다.
    ```bash
    ./run.sh
    ```
    서버가 정상적으로 실행되면 GPU에 모델이 로드되고 요청을 받을 준비가 완료됩니다.

 ### 2. 클라이언트 서버 (FastAPI) 구동 방법

클라이언트 서버는 RAG 파이프라인과 API를 담당하며, Python 가상환경에서 실행됩니다.

1.  **`client` 디렉토리로 이동**:
    새로운 터미널을 열고 `client` 디렉토리로 이동합니다.
    ```bash
    cd client/
    ```

2.  **가상환경 생성 및 활성화**:
    ```bash
    conda create -n sl_assistant python=3.10
    conda activate sl_assistant
    ```
 

3.  **필요 패키지 설치**:
    `requirements.txt` 파일에 명시된 모든 Python 라이브러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

4.  **RAG 데이터베이스 생성**:
    클라이언트 서버를 실행하기 전에, RAG가 참조할 벡터 데이터베이스를 먼저 생성해야 합니다. `scripts/ingest.py` 스크립트를 실행합니다.
    ```bash
    # python -m 옵션을 사용하여 프로젝트 루트에서 모듈로 실행
    python -m scripts.ingest
    ```
    이 스크립트는 `test/` 폴더의 코드를 읽어 `vector_stores/` 디렉토리 안에 `naive_db`와 `raptor_db`를 생성합니다.

5.  **클라이언트 서버 실행**:
    Uvicorn을 사용하여 FastAPI 애플리케이션 서버를 시작합니다.
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    서버가 `http://localhost:8000`에서 실행됩니다.

---
## 📖 사용 방법

클라이언트 서버가 실행되면, `/api/v1/chat` 엔드포인트를 통해 코딩 어시스턴트 기능을 사용할 수 있습니다.

### 1. 대화형 API 문서 (Swagger UI) 사용

1.  웹 브라우저에서 **`http://localhost:8000/docs`** 로 접속합니다.
2.  `POST /api/v1/chat` 엔드포인트를 열고 `Try it out` 버튼을 클릭합니다.
3.  `Request body`에 원하는 질문(`user_prompt`)과 RAG 전략(`rag_strategy`)을 입력하고 `Execute` 버튼을 눌러 결과를 확인합니다.

### 2. `curl`을 이용한 직접 호출

터미널에서 `curl`을 사용하여 API를 직접 호출할 수도 있습니다.

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/chat' \
  -H 'Content-Type: application/json' \
  -d '{
    "user_prompt": "Explain the difference between the C++ and Python versions of the accelerate function.",
    "rag_strategy": "raptor_scoped"
  }'