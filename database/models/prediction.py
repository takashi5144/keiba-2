"""
予測関連のデータベースモデル
"""
from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, DECIMAL, JSON, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from .base import Base

class Prediction(Base):
    """予測結果テーブル"""
    __tablename__ = "predictions"
    
    prediction_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String(20), ForeignKey("races.race_id"), nullable=False)
    horse_id = Column(String(20), ForeignKey("horses.horse_id"), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # 予測確率
    win_probability = Column(DECIMAL(5, 4))
    place_probability = Column(DECIMAL(5, 4))
    show_probability = Column(DECIMAL(5, 4))
    
    # 期待値
    win_expected_value = Column(DECIMAL(6, 3))
    place_expected_value = Column(DECIMAL(6, 3))
    
    # 予測スコア（各モデル）
    lightgbm_score = Column(Float)
    neural_network_score = Column(Float)
    xgboost_score = Column(Float)
    ensemble_score = Column(Float)
    
    # 特徴量の重要度
    feature_importance = Column(JSON)
    
    # 予測時のオッズ
    predicted_odds = Column(DECIMAL(6, 2))
    
    # 信頼度
    confidence_score = Column(DECIMAL(3, 2))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # リレーション
    race = relationship("Race", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(race={self.race_id}, horse={self.horse_id}, win_prob={self.win_probability})>"

class BettingRecommendation(Base):
    """賭け推奨テーブル"""
    __tablename__ = "betting_recommendations"
    
    recommendation_id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(String(20), ForeignKey("races.race_id"), nullable=False)
    bet_type = Column(String(20), nullable=False)  # win, place, exacta, quinella, trifecta
    
    # 推奨内容
    selections = Column(JSON)  # {"horses": ["001", "002"], "box": false}
    
    # 賭け金計算
    kelly_fraction = Column(DECIMAL(4, 3))
    recommended_percentage = Column(DECIMAL(4, 3))
    
    # 期待値
    expected_value = Column(DECIMAL(6, 3))
    expected_return = Column(DECIMAL(8, 2))
    
    # リスク評価
    risk_level = Column(String(20))  # low, medium, high
    confidence_score = Column(DECIMAL(3, 2))
    
    created_at = Column(DateTime, default=datetime.utcnow)

class ModelPerformance(Base):
    """モデルパフォーマンス記録テーブル"""
    __tablename__ = "model_performance"
    
    performance_id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(50), nullable=False)
    evaluation_date = Column(Date, nullable=False)
    
    # パフォーマンス指標
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    accuracy = Column(DECIMAL(5, 4))
    
    # 的中率
    win_hit_rate = Column(DECIMAL(5, 4))
    place_hit_rate = Column(DECIMAL(5, 4))
    
    # 収益性
    total_bets = Column(DECIMAL(10, 2))
    total_returns = Column(DECIMAL(10, 2))
    roi = Column(DECIMAL(6, 3))
    
    # リスク指標
    sharpe_ratio = Column(DECIMAL(5, 3))
    max_drawdown = Column(DECIMAL(5, 4))
    
    # 詳細メトリクス
    metrics_json = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)