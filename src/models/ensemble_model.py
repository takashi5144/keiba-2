"""
LightGBMとニューラルネットワークのアンサンブルモデル
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Dict, List
import joblib
from pathlib import Path

class HorseRacingEnsembleModel:
    """競馬予測のためのアンサンブルモデル"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.lgb_model = None
        self.nn_model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.ensemble_weights = config['ensemble_weights']
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """モデルを訓練"""
        # データの前処理
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # LightGBMの訓練
        lgb_metrics = self._train_lightgbm(X_train, y_train, X_val, y_val)
        
        # ニューラルネットワークの訓練
        nn_metrics = self._train_neural_network(X_train_scaled, y_train, 
                                               X_val_scaled, y_val)
        
        return {
            'lightgbm_metrics': lgb_metrics,
            'neural_network_metrics': nn_metrics,
            'ensemble_metrics': self._evaluate_ensemble(X_val, y_val)
        }
    
    def _train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """LightGBMモデルを訓練"""
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        self.lgb_model = lgb.train(
            self.config['lightgbm_params'],
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'val'],
            num_boost_round=1000,
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ]
        )
        
        # 特徴量重要度を保存
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.lgb_model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # 予測と評価
        y_pred = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration)
        
        return {
            'rmse': np.sqrt(np.mean((y_val - y_pred) ** 2)),
            'mae': np.mean(np.abs(y_val - y_pred)),
            'best_iteration': self.lgb_model.best_iteration
        }
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: pd.Series,
                            X_val: np.ndarray, y_val: pd.Series) -> Dict:
        """ニューラルネットワークを訓練"""
        nn_config = self.config['neural_network']
        
        # モデル構築
        self.nn_model = keras.Sequential()
        
        # 入力層
        self.nn_model.add(keras.layers.Input(shape=(X_train.shape[1],)))
        
        # 隠れ層
        for i, units in enumerate(nn_config['layers']):
            self.nn_model.add(keras.layers.Dense(
                units, 
                activation=nn_config['activation']
            ))
            self.nn_model.add(keras.layers.Dropout(nn_config['dropout_rate']))
            
        # 出力層
        self.nn_model.add(keras.layers.Dense(1))
        
        # コンパイル
        self.nn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # 訓練
        history = self.nn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=nn_config['batch_size'],
            epochs=nn_config['epochs'],
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=0
        )
        
        # 評価
        val_loss, val_mae = self.nn_model.evaluate(X_val, y_val, verbose=0)
        
        return {
            'val_loss': val_loss,
            'val_mae': val_mae,
            'best_epoch': len(history.history['loss']) - 10  # Early stoppingを考慮
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """アンサンブル予測を実行"""
        # LightGBM予測
        lgb_pred = self.lgb_model.predict(X, num_iteration=self.lgb_model.best_iteration)
        
        # ニューラルネットワーク予測
        X_scaled = self.scaler.transform(X)
        nn_pred = self.nn_model.predict(X_scaled, verbose=0).flatten()
        
        # アンサンブル
        ensemble_pred = (
            self.ensemble_weights['lightgbm'] * lgb_pred +
            self.ensemble_weights['neural_network'] * nn_pred
        )
        
        return ensemble_pred
    
    def predict_proba(self, X: pd.DataFrame, n_horses: int) -> np.ndarray:
        """各馬の勝率を予測（ソフトマックス変換）"""
        scores = self.predict(X)
        
        # レースごとにソフトマックス変換
        probabilities = []
        for i in range(0, len(scores), n_horses):
            race_scores = scores[i:i+n_horses]
            # 数値安定性のため最大値を引く
            exp_scores = np.exp(race_scores - np.max(race_scores))
            race_probs = exp_scores / np.sum(exp_scores)
            probabilities.extend(race_probs)
            
        return np.array(probabilities)
    
    def _evaluate_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """アンサンブルモデルを評価"""
        y_pred = self.predict(X_val)
        
        return {
            'rmse': np.sqrt(np.mean((y_val - y_pred) ** 2)),
            'mae': np.mean(np.abs(y_val - y_pred)),
            'correlation': np.corrcoef(y_val, y_pred)[0, 1]
        }
    
    def save_model(self, model_dir: Path):
        """モデルを保存"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # LightGBMモデル
        self.lgb_model.save_model(str(model_dir / 'lightgbm_model.txt'))
        
        # ニューラルネットワーク
        self.nn_model.save(str(model_dir / 'neural_network'))
        
        # スケーラーと特徴量重要度
        joblib.dump(self.scaler, model_dir / 'scaler.pkl')
        self.feature_importance.to_csv(model_dir / 'feature_importance.csv', index=False)
        
    def load_model(self, model_dir: Path):
        """モデルを読み込み"""
        # LightGBMモデル
        self.lgb_model = lgb.Booster(model_file=str(model_dir / 'lightgbm_model.txt'))
        
        # ニューラルネットワーク
        self.nn_model = keras.models.load_model(str(model_dir / 'neural_network'))
        
        # スケーラー
        self.scaler = joblib.load(model_dir / 'scaler.pkl')
        
        # 特徴量重要度
        self.feature_importance = pd.read_csv(model_dir / 'feature_importance.csv')