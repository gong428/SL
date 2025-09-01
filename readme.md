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
    (옵션 A: Conda 사용 시)
    ```bash
    conda create -n sl_assistant python=3.10
    conda activate sl_assistant
    ```
    (옵션 B: venv 사용 시)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
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
