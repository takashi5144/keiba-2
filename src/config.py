"""
高精度競馬予測システムの設定
"""
import os
from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# データパス
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"

# モデル設定
MODEL_CONFIG = {
    "ensemble_weights": {
        "lightgbm": 0.7,
        "neural_network": 0.3
    },
    "lightgbm_params": {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1
    },
    "neural_network": {
        "layers": [96, 64, 32],
        "activation": "relu",
        "dropout_rate": 0.3,
        "batch_size": 100,
        "epochs": 40
    }
}

# 特徴量設定
FEATURE_CONFIG = {
    "time_windows": [3, 5, 10],  # 直近レース数
    "speed_index_normalize": True,
    "class_rating": True,
    "sectional_analysis": True
}

# 資金管理設定
MONEY_MANAGEMENT = {
    "kelly_fraction": 0.25,  # フラクショナル・ケリー
    "max_bet_percentage": 0.025,  # 単一レースへの最大賭け金割合
    "min_expected_value": 1.05,  # 最小期待値
    "confidence_threshold": 0.7  # 予測信頼度の閾値
}

# スクレイピング設定
SCRAPING_CONFIG = {
    "base_url": "https://db.netkeiba.com",
    "delay_range": (1, 5),  # リクエスト間の遅延（秒）
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "encoding": "EUC-JP"
}

# API設定
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "workers": 4
}

# 評価指標
EVALUATION_METRICS = {
    "roi_target": 0.05,  # 目標ROI（5%）
    "sharpe_ratio_target": 1.0,
    "max_drawdown_limit": 0.20,
    "confidence_interval": 0.95
}