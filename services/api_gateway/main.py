"""
FastAPI メインアプリケーション
"""
from fastapi import FastAPI, HTTPException, Depends, Security, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from typing import List, Optional
from datetime import date, datetime
from loguru import logger

from .models import (
    RacePredictionResponse, 
    BettingRecommendationResponse,
    ModelPerformanceResponse,
    RaceListResponse
)
from services.prediction_engine.models import HorseRacingEnsembleModel
from services.money_management.betting import BettingStrategy
from services.data_collector.tasks import collect_race_data, collect_daily_races
from database.models.base import get_db, init_db
from database.models.race import Race
from database.models.prediction import Prediction, ModelPerformance

# 環境変数
API_KEY = os.getenv("API_KEY", "your-secure-api-key")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# APIキー認証
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# モデルのグローバルインスタンス
prediction_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # 起動時
    logger.info("Starting HorseRacingAI API...")
    
    # データベース初期化
    init_db()
    
    # モデルを読み込み
    global prediction_model
    prediction_model = HorseRacingEnsembleModel()
    try:
        prediction_model.load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load models: {e}")
    
    yield
    
    # 終了時
    if prediction_model:
        prediction_model.close()
    logger.info("Shutting down HorseRacingAI API...")

# FastAPIアプリケーション
app = FastAPI(
    title="HorseRacingAI API",
    description="高精度競馬予測システムAPI",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=dict)
async def root():
    """APIルート"""
    return {
        "message": "HorseRacingAI API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/docs": "API documentation",
            "/api/v1/races": "List races",
            "/api/v1/predictions/{race_id}": "Get race predictions",
            "/api/v1/betting/{race_id}": "Get betting recommendations",
            "/api/v1/performance": "Get model performance",
            "/api/v1/collect/{race_id}": "Collect race data"
        }
    }

@app.get("/api/v1/races", response_model=RaceListResponse)
async def get_races(
    target_date: Optional[date] = None,
    track: Optional[str] = None,
    grade: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """レース一覧を取得"""
    query = db.query(Race)
    
    if target_date:
        query = query.filter(Race.race_date == target_date)
    if track:
        query = query.filter(Race.track == track)
    if grade:
        query = query.filter(Race.grade == grade)
    
    races = query.order_by(Race.race_date.desc(), Race.race_number).limit(limit).all()
    
    return {
        "count": len(races),
        "races": [
            {
                "race_id": race.race_id,
                "race_name": race.race_name,
                "race_date": race.race_date,
                "track": race.track,
                "distance": race.distance,
                "surface": race.surface,
                "grade": race.grade,
                "start_time": race.start_time
            }
            for race in races
        ]
    }

@app.get("/api/v1/predictions/{race_id}", response_model=RacePredictionResponse)
async def get_predictions(
    race_id: str,
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """レースの予測結果を取得"""
    # レース情報を確認
    race = db.query(Race).filter_by(race_id=race_id).first()
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
    
    # 既存の予測を確認
    existing_predictions = db.query(Prediction).filter_by(
        race_id=race_id,
        model_version=prediction_model.model_version
    ).all()
    
    if not existing_predictions:
        # 新規予測を実行
        if not prediction_model:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        predictions_df = prediction_model.predict(race_id)
        
        if predictions_df.empty:
            raise HTTPException(status_code=500, detail="Prediction failed")
    
    # 予測結果を取得
    predictions = db.query(Prediction).filter_by(
        race_id=race_id,
        model_version=prediction_model.model_version
    ).order_by(Prediction.win_probability.desc()).all()
    
    return {
        "race_id": race_id,
        "race_info": {
            "race_name": race.race_name,
            "race_date": race.race_date,
            "track": race.track,
            "distance": race.distance,
            "surface": race.surface,
            "weather": race.weather,
            "track_condition": race.track_condition
        },
        "model_version": prediction_model.model_version,
        "predictions": [
            {
                "horse_id": pred.horse_id,
                "win_probability": float(pred.win_probability),
                "place_probability": float(pred.place_probability),
                "show_probability": float(pred.show_probability),
                "confidence_score": float(pred.confidence_score),
                "ensemble_score": float(pred.ensemble_score)
            }
            for pred in predictions
        ],
        "timestamp": datetime.now()
    }

@app.get("/api/v1/betting/{race_id}", response_model=BettingRecommendationResponse)
async def get_betting_recommendations(
    race_id: str,
    bankroll: float = 100000,
    risk_level: str = "moderate",
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """賭け推奨を取得"""
    # 予測結果を取得
    predictions = db.query(Prediction).filter_by(
        race_id=race_id,
        model_version=prediction_model.model_version
    ).all()
    
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found for this race")
    
    # 賭け戦略を計算
    betting_strategy = BettingStrategy()
    recommendations = betting_strategy.calculate_optimal_bets(
        predictions, bankroll, risk_level
    )
    
    return recommendations

@app.get("/api/v1/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    days: int = 30,
    api_key: str = Depends(verify_api_key),
    db = Depends(get_db)
):
    """モデルパフォーマンスを取得"""
    cutoff_date = datetime.now().date() - timedelta(days=days)
    
    performances = db.query(ModelPerformance).filter(
        ModelPerformance.evaluation_date >= cutoff_date,
        ModelPerformance.model_version == prediction_model.model_version
    ).order_by(ModelPerformance.evaluation_date.desc()).all()
    
    if not performances:
        return {
            "model_version": prediction_model.model_version,
            "period_days": days,
            "metrics": {
                "average_accuracy": 0,
                "average_roi": 0,
                "total_predictions": 0,
                "sharpe_ratio": 0
            }
        }
    
    # 集計
    total_predictions = sum(p.total_predictions for p in performances)
    total_correct = sum(p.correct_predictions for p in performances)
    average_roi = np.mean([p.roi for p in performances if p.roi])
    
    return {
        "model_version": prediction_model.model_version,
        "period_days": days,
        "metrics": {
            "average_accuracy": total_correct / total_predictions if total_predictions > 0 else 0,
            "average_roi": float(average_roi) if average_roi else 0,
            "total_predictions": total_predictions,
            "sharpe_ratio": performances[0].sharpe_ratio if performances[0].sharpe_ratio else 0
        },
        "daily_performance": [
            {
                "date": p.evaluation_date,
                "accuracy": float(p.accuracy),
                "roi": float(p.roi) if p.roi else 0,
                "predictions": p.total_predictions
            }
            for p in performances
        ]
    }

@app.post("/api/v1/collect/{race_id}")
async def trigger_data_collection(
    race_id: str,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """レースデータ収集をトリガー"""
    # バックグラウンドでデータ収集を実行
    background_tasks.add_task(collect_race_data.delay, race_id)
    
    return {
        "message": f"Data collection triggered for race {race_id}",
        "status": "queued"
    }

@app.post("/api/v1/collect/daily")
async def trigger_daily_collection(
    target_date: Optional[date] = None,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """日次データ収集をトリガー"""
    date_str = target_date.strftime("%Y-%m-%d") if target_date else None
    
    # バックグラウンドでデータ収集を実行
    background_tasks.add_task(collect_daily_races.delay, date_str)
    
    return {
        "message": f"Daily data collection triggered for {date_str or 'today'}",
        "status": "queued"
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """一般的な例外ハンドラー"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.api_gateway.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True
    )