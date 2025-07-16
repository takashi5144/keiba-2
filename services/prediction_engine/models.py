"""
アンサンブル機械学習モデル
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import joblib
from pathlib import Path
from datetime import datetime
from loguru import logger

from services.feature_engineering.features import FeatureEngineering
from database.models.base import SessionLocal
from database.models.prediction import Prediction, ModelPerformance

class ModelConfig:
    """モデル設定"""
    LIGHTGBM_PARAMS = {
        'objective': 'multiclass',
        'num_class': 18,  # 最大18頭
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'max_depth': -1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'min_child_samples': 20,
        'random_state': 42
    }
    
    XGBOOST_PARAMS = {
        'objective': 'multi:softprob',
        'num_class': 18,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'mlogloss',
        'seed': 42,
        'nthread': -1
    }
    
    NEURAL_NETWORK_CONFIG = {
        'layers': [256, 128, 64],
        'dropout_rates': [0.3, 0.3, 0.2],
        'activation': 'relu',
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'batch_size': 128,
        'epochs': 100,
        'early_stopping_patience': 10
    }
    
    ENSEMBLE_WEIGHTS = {
        'lightgbm': 0.4,
        'xgboost': 0.3,
        'neural_network': 0.3
    }

class HorseRacingEnsembleModel:
    """競馬予測アンサンブルモデル"""
    
    def __init__(self, model_version: str = "v1.0.0"):
        self.model_version = model_version
        self.config = ModelConfig()
        self.feature_engineering = FeatureEngineering()
        self.db = SessionLocal()
        
        # モデル
        self.lgb_model = None
        self.xgb_model = None
        self.nn_model = None
        
        # 前処理
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        # モデルパス
        self.model_dir = Path("data/models") / model_version
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self, start_date: str, end_date: str, validation_split: float = 0.2):
        """モデルを訓練"""
        logger.info(f"Training models from {start_date} to {end_date}")
        
        # データを準備
        X_train, y_train, X_val, y_val = self._prepare_training_data(
            start_date, end_date, validation_split
        )
        
        if X_train is None or len(X_train) == 0:
            logger.error("No training data available")
            return
        
        # 特徴量を保存
        self.feature_columns = list(X_train.columns)
        
        # 各モデルを訓練
        logger.info("Training LightGBM model...")
        self._train_lightgbm(X_train, y_train, X_val, y_val)
        
        logger.info("Training XGBoost model...")
        self._train_xgboost(X_train, y_train, X_val, y_val)
        
        logger.info("Training Neural Network model...")
        self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # モデルを保存
        self.save_models()
        
        # パフォーマンスを評価して記録
        self._evaluate_and_record_performance(X_val, y_val)
    
    def _prepare_training_data(self, start_date: str, end_date: str, 
                              validation_split: float) -> Tuple:
        """訓練データを準備"""
        from database.models.race import Race, RaceResult
        
        # 対象期間のレースを取得
        races = self.db.query(Race).filter(
            Race.race_date >= start_date,
            Race.race_date <= end_date
        ).order_by(Race.race_date).all()
        
        all_features = []
        all_labels = []
        
        for race in races:
            # レースの特徴量を生成
            features_df = self.feature_engineering.create_features_for_race(race.race_id)
            
            if features_df.empty:
                continue
            
            # レース結果を取得
            results = self.db.query(RaceResult).filter_by(
                race_id=race.race_id
            ).order_by(RaceResult.horse_id).all()
            
            # ラベル（着順）を作成
            labels = {}
            for result in results:
                if result.finish_position and result.finish_position > 0:
                    labels[result.horse_id] = result.finish_position - 1  # 0-indexed
            
            # 特徴量とラベルを対応付け
            for _, row in features_df.iterrows():
                if row['horse_id'] in labels:
                    all_features.append(row.drop('horse_id'))
                    all_labels.append(labels[row['horse_id']])
        
        if not all_features:
            return None, None, None, None
        
        # データフレームに変換
        X = pd.DataFrame(all_features)
        y = np.array(all_labels)
        
        # 時系列分割
        split_index = int(len(X) * (1 - validation_split))
        X_train = X[:split_index]
        y_train = y[:split_index]
        X_val = X[split_index:]
        y_val = y[split_index:]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: np.ndarray,
                       X_val: pd.DataFrame, y_val: np.ndarray):
        """LightGBMモデルを訓練"""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.lgb_model = lgb.train(
            self.config.LIGHTGBM_PARAMS,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        # 特徴量重要度を保存
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.lgb_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(self.model_dir / "feature_importance_lgb.csv", index=False)
    
    def _train_xgboost(self, X_train: pd.DataFrame, y_train: np.ndarray,
                      X_val: pd.DataFrame, y_val: np.ndarray):
        """XGBoostモデルを訓練"""
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        self.xgb_model = xgb.train(
            self.config.XGBOOST_PARAMS,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
    
    def _train_neural_network(self, X_train: pd.DataFrame, y_train: np.ndarray,
                            X_val: pd.DataFrame, y_val: np.ndarray):
        """ニューラルネットワークモデルを訓練"""
        # データを正規化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # One-hot encoding for labels
        num_classes = self.config.LIGHTGBM_PARAMS['num_class']
        y_train_onehot = keras.utils.to_categorical(y_train, num_classes)
        y_val_onehot = keras.utils.to_categorical(y_val, num_classes)
        
        # モデル構築
        self.nn_model = self._build_neural_network(X_train.shape[1], num_classes)
        
        # コールバック
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=self.config.NEURAL_NETWORK_CONFIG['early_stopping_patience'],
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
            keras.callbacks.ModelCheckpoint(
                str(self.model_dir / "nn_model_best.h5"),
                save_best_only=True
            )
        ]
        
        # 訓練
        history = self.nn_model.fit(
            X_train_scaled, y_train_onehot,
            validation_data=(X_val_scaled, y_val_onehot),
            epochs=self.config.NEURAL_NETWORK_CONFIG['epochs'],
            batch_size=self.config.NEURAL_NETWORK_CONFIG['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # 訓練履歴を保存
        pd.DataFrame(history.history).to_csv(
            self.model_dir / "nn_training_history.csv", index=False
        )
    
    def _build_neural_network(self, input_dim: int, num_classes: int) -> keras.Model:
        """ニューラルネットワークを構築"""
        model = keras.Sequential()
        
        # 入力層
        model.add(keras.layers.Input(shape=(input_dim,)))
        
        # 隠れ層
        nn_config = self.config.NEURAL_NETWORK_CONFIG
        for i, (units, dropout_rate) in enumerate(zip(
            nn_config['layers'], 
            nn_config['dropout_rates']
        )):
            model.add(keras.layers.Dense(units, activation=nn_config['activation']))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(dropout_rate))
        
        # 出力層
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        # コンパイル
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        return model
    
    def predict(self, race_id: str) -> pd.DataFrame:
        """レースの予測を実行"""
        logger.info(f"Predicting race: {race_id}")
        
        # 特徴量を生成
        features_df = self.feature_engineering.create_features_for_race(race_id)
        
        if features_df.empty:
            logger.error(f"No features generated for race: {race_id}")
            return pd.DataFrame()
        
        # 必要なカラムのみ選択
        X = features_df[self.feature_columns]
        
        # 各モデルで予測
        predictions = {}
        
        # LightGBM
        lgb_probs = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)
        predictions['lightgbm'] = lgb_probs
        
        # XGBoost
        dtest = xgb.DMatrix(X)
        xgb_probs = self.xgb_model.predict(dtest)
        predictions['xgboost'] = xgb_probs
        
        # Neural Network
        X_scaled = self.scaler.transform(X)
        nn_probs = self.nn_model.predict(X_scaled, verbose=0)
        predictions['neural_network'] = nn_probs
        
        # アンサンブル予測
        ensemble_probs = self._ensemble_predictions(predictions)
        
        # 結果をデータフレームに整理
        results_df = features_df[['horse_id', 'post_position']].copy()
        
        # 勝率（1着になる確率）
        results_df['win_probability'] = ensemble_probs[:, 0]
        
        # 連対率（2着以内）
        results_df['place_probability'] = ensemble_probs[:, :2].sum(axis=1)
        
        # 複勝率（3着以内）
        results_df['show_probability'] = ensemble_probs[:, :3].sum(axis=1)
        
        # 各モデルのスコア
        results_df['lightgbm_score'] = predictions['lightgbm'][:, 0]
        results_df['xgboost_score'] = predictions['xgboost'][:, 0]
        results_df['neural_network_score'] = predictions['neural_network'][:, 0]
        
        # 予測をデータベースに保存
        self._save_predictions(race_id, results_df)
        
        return results_df.sort_values('win_probability', ascending=False)
    
    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """アンサンブル予測を計算"""
        weights = self.config.ENSEMBLE_WEIGHTS
        
        ensemble_probs = (
            weights['lightgbm'] * predictions['lightgbm'] +
            weights['xgboost'] * predictions['xgboost'] +
            weights['neural_network'] * predictions['neural_network']
        )
        
        # 正規化（合計が1になるように）
        ensemble_probs = ensemble_probs / ensemble_probs.sum(axis=1, keepdims=True)
        
        return ensemble_probs
    
    def _save_predictions(self, race_id: str, predictions_df: pd.DataFrame):
        """予測結果をデータベースに保存"""
        for _, row in predictions_df.iterrows():
            prediction = Prediction(
                race_id=race_id,
                horse_id=row['horse_id'],
                model_version=self.model_version,
                win_probability=float(row['win_probability']),
                place_probability=float(row['place_probability']),
                show_probability=float(row['show_probability']),
                lightgbm_score=float(row['lightgbm_score']),
                xgboost_score=float(row['xgboost_score']),
                neural_network_score=float(row['neural_network_score']),
                ensemble_score=float(row['win_probability']),
                confidence_score=self._calculate_confidence(row)
            )
            self.db.add(prediction)
        
        self.db.commit()
    
    def _calculate_confidence(self, prediction_row: pd.Series) -> float:
        """予測の信頼度を計算"""
        # モデル間の予測の一致度
        scores = [
            prediction_row['lightgbm_score'],
            prediction_row['xgboost_score'],
            prediction_row['neural_network_score']
        ]
        
        # 標準偏差が小さいほど信頼度が高い
        std_dev = np.std(scores)
        confidence = 1.0 - min(std_dev * 2, 0.5)  # 0.5〜1.0の範囲
        
        return confidence
    
    def save_models(self):
        """モデルを保存"""
        # LightGBM
        self.lgb_model.save_model(str(self.model_dir / "lightgbm_model.txt"))
        
        # XGBoost
        self.xgb_model.save_model(str(self.model_dir / "xgboost_model.json"))
        
        # Neural Network
        self.nn_model.save(str(self.model_dir / "neural_network_model.h5"))
        
        # Scaler
        joblib.dump(self.scaler, self.model_dir / "scaler.pkl")
        
        # Feature columns
        joblib.dump(self.feature_columns, self.model_dir / "feature_columns.pkl")
        
        logger.info(f"Models saved to {self.model_dir}")
    
    def load_models(self):
        """モデルを読み込み"""
        # LightGBM
        self.lgb_model = lgb.Booster(model_file=str(self.model_dir / "lightgbm_model.txt"))
        
        # XGBoost
        self.xgb_model = xgb.Booster()
        self.xgb_model.load_model(str(self.model_dir / "xgboost_model.json"))
        
        # Neural Network
        self.nn_model = keras.models.load_model(str(self.model_dir / "neural_network_model.h5"))
        
        # Scaler
        self.scaler = joblib.load(self.model_dir / "scaler.pkl")
        
        # Feature columns
        self.feature_columns = joblib.load(self.model_dir / "feature_columns.pkl")
        
        logger.info(f"Models loaded from {self.model_dir}")
    
    def _evaluate_and_record_performance(self, X_val: pd.DataFrame, y_val: np.ndarray):
        """モデルパフォーマンスを評価して記録"""
        # 予測を実行
        predictions = {}
        
        # 各モデルで予測
        lgb_probs = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration)
        predictions['lightgbm'] = lgb_probs
        
        dval = xgb.DMatrix(X_val)
        xgb_probs = self.xgb_model.predict(dval)
        predictions['xgboost'] = xgb_probs
        
        X_val_scaled = self.scaler.transform(X_val)
        nn_probs = self.nn_model.predict(X_val_scaled, verbose=0)
        predictions['neural_network'] = nn_probs
        
        # アンサンブル予測
        ensemble_probs = self._ensemble_predictions(predictions)
        
        # 予測クラス（最も確率の高い着順）
        y_pred = np.argmax(ensemble_probs, axis=1)
        
        # メトリクスを計算
        accuracy = accuracy_score(y_val, y_pred)
        logloss = log_loss(y_val, ensemble_probs)
        
        # Top-3精度（複勝的中率に相当）
        top3_accuracy = np.mean([
            1 if true in pred else 0 
            for true, pred in zip(y_val, np.argsort(ensemble_probs, axis=1)[:, -3:])
        ])
        
        # パフォーマンスを記録
        performance = ModelPerformance(
            model_version=self.model_version,
            evaluation_date=datetime.now().date(),
            total_predictions=len(y_val),
            correct_predictions=int(accuracy * len(y_val)),
            accuracy=float(accuracy),
            metrics_json={
                'log_loss': float(logloss),
                'top3_accuracy': float(top3_accuracy),
                'lightgbm_iterations': self.lgb_model.best_iteration,
                'xgboost_iterations': self.xgb_model.best_iteration
            }
        )
        
        self.db.add(performance)
        self.db.commit()
        
        logger.info(f"Model performance - Accuracy: {accuracy:.4f}, Top-3: {top3_accuracy:.4f}")
    
    def close(self):
        """リソースをクリーンアップ"""
        self.feature_engineering.close()
        self.db.close()