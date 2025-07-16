"""
競馬予測のための特徴量エンジニアリング
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class HorseFeatureEngineering:
    """馬の特徴量を生成するクラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.time_windows = config.get('time_windows', [3, 5, 10])
        
    def create_time_based_features(self, horse_history: pd.DataFrame) -> pd.DataFrame:
        """時間ベースの特徴量を生成"""
        features = pd.DataFrame()
        
        for window in self.time_windows:
            # 直近N戦の成績
            recent_races = horse_history.head(window)
            
            # 平均着順
            features[f'avg_finish_last_{window}'] = recent_races['finish_position'].mean()
            
            # 勝率
            features[f'win_rate_last_{window}'] = (recent_races['finish_position'] == 1).mean()
            
            # 連対率
            features[f'place_rate_last_{window}'] = (recent_races['finish_position'] <= 2).mean()
            
            # 複勝率
            features[f'show_rate_last_{window}'] = (recent_races['finish_position'] <= 3).mean()
            
            # スピード指数の平均と標準偏差
            if 'speed_index' in recent_races.columns:
                features[f'avg_speed_index_last_{window}'] = recent_races['speed_index'].mean()
                features[f'std_speed_index_last_{window}'] = recent_races['speed_index'].std()
                
            # 賞金獲得額
            if 'prize_money' in recent_races.columns:
                features[f'total_prize_last_{window}'] = recent_races['prize_money'].sum()
                
        return features
    
    def create_performance_indicators(self, race_data: pd.DataFrame) -> pd.DataFrame:
        """パフォーマンス指標を生成"""
        features = pd.DataFrame()
        
        # 正規化スピード指数
        if self.config.get('speed_index_normalize', True):
            features['normalized_speed_index'] = self._normalize_speed_index(
                race_data['time'], 
                race_data['distance'],
                race_data['track_condition']
            )
        
        # クラスレーティング
        if self.config.get('class_rating', True):
            features['class_rating'] = self._calculate_class_rating(
                race_data['race_class'],
                race_data['finish_position']
            )
        
        # セクショナルタイム分析
        if self.config.get('sectional_analysis', True) and 'sectional_times' in race_data.columns:
            sectional_features = self._analyze_sectional_times(race_data['sectional_times'])
            features = pd.concat([features, sectional_features], axis=1)
            
        return features
    
    def create_interaction_features(self, horse_data: pd.DataFrame, 
                                  jockey_data: pd.DataFrame,
                                  trainer_data: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量を生成"""
        features = pd.DataFrame()
        
        # 馬と騎手の相性
        features['horse_jockey_win_rate'] = self._calculate_combination_stats(
            horse_data, jockey_data, 'win_rate'
        )
        
        # 調教師の専門性
        features['trainer_distance_expertise'] = self._calculate_trainer_expertise(
            trainer_data, horse_data['distance']
        )
        
        # 血統と馬場の相関
        if 'pedigree' in horse_data.columns:
            features['bloodline_track_affinity'] = self._calculate_bloodline_affinity(
                horse_data['pedigree'], horse_data['track_type']
            )
            
        return features
    
    def _normalize_speed_index(self, time: float, distance: int, 
                             track_condition: str) -> float:
        """スピード指数を正規化"""
        # 基準タイム（1000mあたり）
        base_time_per_1000m = 60.0
        
        # 馬場状態による補正
        condition_adjustments = {
            '良': 0.0,
            '稍重': 0.5,
            '重': 1.0,
            '不良': 2.0
        }
        
        adjustment = condition_adjustments.get(track_condition, 0.0)
        adjusted_time = time - adjustment
        
        # スピード指数計算
        speed_index = (base_time_per_1000m * distance / 1000) / adjusted_time * 100
        
        return speed_index
    
    def _calculate_class_rating(self, race_class: str, 
                               finish_position: int) -> float:
        """クラスレーティングを計算"""
        # レースクラスの重み
        class_weights = {
            'G1': 1.0,
            'G2': 0.9,
            'G3': 0.8,
            'オープン': 0.7,
            '3勝': 0.6,
            '2勝': 0.5,
            '1勝': 0.4,
            '未勝利': 0.3,
            '新馬': 0.2
        }
        
        base_weight = class_weights.get(race_class, 0.5)
        
        # 着順による調整
        position_factor = 1.0 - (finish_position - 1) * 0.05
        
        return base_weight * position_factor
    
    def _analyze_sectional_times(self, sectional_times: List[float]) -> pd.DataFrame:
        """セクショナルタイムを分析"""
        features = pd.DataFrame()
        
        if len(sectional_times) >= 3:
            # 前半・中盤・後半のペース
            features['early_pace'] = np.mean(sectional_times[:len(sectional_times)//3])
            features['mid_pace'] = np.mean(sectional_times[len(sectional_times)//3:2*len(sectional_times)//3])
            features['late_pace'] = np.mean(sectional_times[2*len(sectional_times)//3:])
            
            # ペース変化
            features['pace_variation'] = np.std(sectional_times)
            features['acceleration'] = features['early_pace'] - features['late_pace']
            
        return features
    
    def _calculate_combination_stats(self, horse_data: pd.DataFrame,
                                   jockey_data: pd.DataFrame,
                                   stat_type: str) -> float:
        """馬と騎手の組み合わせ統計を計算"""
        # 実装の簡略版
        return 0.15  # デフォルト値
    
    def _calculate_trainer_expertise(self, trainer_data: pd.DataFrame,
                                   distance: int) -> float:
        """調教師の距離適性を計算"""
        # 実装の簡略版
        return 0.5  # デフォルト値
    
    def _calculate_bloodline_affinity(self, pedigree: str,
                                    track_type: str) -> float:
        """血統と馬場の親和性を計算"""
        # 実装の簡略版
        return 0.5  # デフォルト値