"""
FastAPI用のPydanticモデル定義
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime, time
from decimal import Decimal

class RaceInfo(BaseModel):
    """レース情報"""
    race_id: str
    race_name: str
    race_date: date
    track: str
    distance: int
    surface: str
    grade: Optional[str]
    weather: Optional[str]
    track_condition: Optional[str]
    start_time: Optional[time]

class HorsePrediction(BaseModel):
    """馬の予測結果"""
    horse_id: str
    horse_name: Optional[str]
    post_position: Optional[int]
    win_probability: float = Field(..., ge=0, le=1)
    place_probability: float = Field(..., ge=0, le=1)
    show_probability: float = Field(..., ge=0, le=1)
    confidence_score: float = Field(..., ge=0, le=1)
    ensemble_score: float
    expected_value: Optional[float]

class RacePredictionResponse(BaseModel):
    """レース予測レスポンス"""
    race_id: str
    race_info: RaceInfo
    model_version: str
    predictions: List[HorsePrediction]
    timestamp: datetime

class BettingRecommendation(BaseModel):
    """賭け推奨"""
    bet_type: str  # win, place, exacta, quinella, trifecta
    selections: List[str]  # 馬ID
    kelly_fraction: float
    recommended_percentage: float
    recommended_amount: float
    expected_value: float
    expected_return: float
    confidence_score: float
    risk_level: str

class BettingRecommendationResponse(BaseModel):
    """賭け推奨レスポンス"""
    race_id: str
    bankroll: float
    risk_level: str
    recommendations: List[BettingRecommendation]
    summary: Dict[str, Any]
    warnings: List[str]
    timestamp: datetime

class DailyPerformance(BaseModel):
    """日次パフォーマンス"""
    date: date
    accuracy: float
    roi: float
    predictions: int
    wins: Optional[int]
    total_bets: Optional[float]
    total_returns: Optional[float]

class ModelPerformanceResponse(BaseModel):
    """モデルパフォーマンスレスポンス"""
    model_version: str
    period_days: int
    metrics: Dict[str, float]
    daily_performance: List[DailyPerformance]

class RaceListItem(BaseModel):
    """レースリストアイテム"""
    race_id: str
    race_name: str
    race_date: date
    track: str
    distance: int
    surface: str
    grade: Optional[str]
    start_time: Optional[time]
    entries_count: Optional[int]

class RaceListResponse(BaseModel):
    """レースリストレスポンス"""
    count: int
    races: List[RaceListItem]

class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    detail: str
    code: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.now)