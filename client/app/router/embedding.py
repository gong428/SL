from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os

# pipeline.py가 아닌 scripts/ingest_codebase.py에서 함수를 임포트합니다.
from scripts.ingest_codebase import run_ingestion_pipeline

router = APIRouter()

class IngestionRequest(BaseModel):
    code_path: str
    vector_store_name: str
    language: str

@router.post("/ingest-codebase", status_code=202)
async def trigger_ingestion(request: IngestionRequest, background_tasks: BackgroundTasks):
    """
    코드베이스 인덱싱 파이프라인을 비동기적으로 실행하는 API 엔드포인트.
    """
    # 입력받은 경로가 실제로 존재하는지 간단히 확인
    if not os.path.isdir(request.code_path):
        raise HTTPException(status_code=404, detail="제공된 코드 경로를 찾을 수 없습니다.")

    print(f"인덱싱 요청 수신: {request.vector_store_name}")
    
    # 인덱싱은 시간이 오래 걸릴 수 있으므로 백그라운드 작업으로 실행합니다.
    background_tasks.add_task(
        run_ingestion_pipeline,
        request.code_path,
        request.vector_store_name,
        request.language
    )
    
    return {"message": f"'{request.vector_store_name}'에 대한 코드베이스 인덱싱 작업이 시작되었습니다."}