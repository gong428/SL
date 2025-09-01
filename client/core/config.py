# 파일: client/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """애플리케이션 설정을 관리하는 클래스입니다."""
    
    # --- 추론 서버 설정 ---
    TRITON_SERVER_URL: str = "http://localhost:8888"
    TRITON_TIMEOUT_SECONDS: float = 300.0
    
    # --- 모델 설정 ---
    # Triton 서버의 model.json에 있는 모델 이름과 동일해야 합니다.
    MODEL_NAME: str = "Nxcode" 
    
    # 클라이언트에서 채팅 템플릿을 적용하기 위해 사용할 토크나이저의 Hugging Face 경로입니다.
    # 보통 서빙되는 모델과 동일합니다.
    TOKENIZER_NAME_OR_PATH: str = "NTQAI/Nxcode-CQ-7B-orpo"

# 설정 객체를 다른 파일에서 쉽게 가져다 쓸 수 있도록 함수를 만듭니다.
def get_settings():
    return Settings()