"""GET /api/history, GET /api/stats, GET /api/patients, DELETE /api/history."""

from typing import List

from fastapi import APIRouter

from web.backend.models import DailyStats, PatientSummary, ScanRecord
from web.backend.services import store

router = APIRouter()


@router.get("/api/history", response_model=List[ScanRecord])
async def get_history(limit: int = 50, offset: int = 0):
    return store.get_history(limit, offset)


@router.get("/api/stats", response_model=DailyStats)
async def get_stats():
    return store.get_daily_stats()


@router.get("/api/patients", response_model=List[PatientSummary])
async def get_patients():
    return store.get_patients()


@router.get("/api/patients/{patient_name}/scans", response_model=List[ScanRecord])
async def get_patient_scans(patient_name: str):
    return store.get_patient_scans(patient_name)


@router.delete("/api/history")
async def clear_history():
    count = store.clear_history()
    return {"deleted": count}
