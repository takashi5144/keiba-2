"""
高度な特徴量エンジニアリング
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger

from database.models.base import SessionLocal
from database.models.race import Race, RaceResult
from database.models.horse import Horse, Jockey, Trainer

class FeatureEngineering:
    """特徴量生成クラス"""
    
    def __init__(self):
        self.db = SessionLocal()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_features_for_race(self, race_id: str) -> pd.DataFrame:
        """レースの全馬の特徴量を生成"""
        logger.info(f"Creating features for race: {race_id}")
        
        # レース情報を取得
        race = self.db.query(Race).filter_by(race_id=race_id).first()
        if not race:
            logger.error(f"Race not found: {race_id}")
            return pd.DataFrame()
        
        # 出走馬を取得
        entries = self.db.query(RaceResult).filter_by(race_id=race_id).all()
        
        features_list = []
        for entry in entries:
            horse_features = self.create_horse_features(
                entry.horse_id, 
                race.race_date,
                race
            )
            if horse_features is not None:
                horse_features['horse_id'] = entry.horse_id
                horse_features['post_position'] = entry.post_position
                features_list.append(horse_features)
        
        if not features_list:
            return pd.DataFrame()
        
        # データフレームに変換
        features_df = pd.DataFrame(features_list)
        
        # レース全体の特徴量を追加
        features_df = self._add_race_level_features(features_df, race)
        
        return features_df
    
    def create_horse_features(self, horse_id: str, race_date: date, 
                            race: Race) -> Optional[Dict]:
        """個別馬の特徴量を生成"""
        horse = self.db.query(Horse).filter_by(horse_id=horse_id).first()
        if not horse:
            return None
        
        features = {}
        
        # 基本特徴量
        features.update(self._create_basic_features(horse, race_date))
        
        # パフォーマンス特徴量
        features.update(self._create_performance_features(horse_id, race_date))
        
        # 距離・馬場適性
        features.update(self._create_track_features(horse_id, race))
        
        # 騎手・調教師特徴量
        features.update(self._create_human_features(horse_id, race_date))
        
        # 血統特徴量
        features.update(self._create_pedigree_features(horse))
        
        # 調子・フォーム特徴量
        features.update(self._create_form_features(horse_id, race_date))
        
        return features
    
    def _create_basic_features(self, horse: Horse, race_date: date) -> Dict:
        """基本特徴量を生成"""
        features = {}
        
        # 馬齢
        if horse.birth_date:
            age_days = (race_date - horse.birth_date).days
            features['horse_age_years'] = age_days / 365.25
            features['horse_age_months'] = age_days / 30.44
        
        # 性別
        features['is_male'] = 1 if horse.sex in ['牡', '騙'] else 0
        features['is_female'] = 1 if horse.sex == '牝' else 0
        features['is_castrated'] = 1 if horse.sex == '騙' else 0
        
        return features
    
    def _create_performance_features(self, horse_id: str, race_date: date) -> Dict:
        """パフォーマンス特徴量を生成"""
        features = {}
        
        # 過去のレース結果を取得
        past_results = self.db.query(RaceResult).join(Race).filter(
            RaceResult.horse_id == horse_id,
            Race.race_date < race_date
        ).order_by(Race.race_date.desc()).limit(20).all()
        
        if not past_results:
            # デフォルト値を設定
            features.update({
                'career_starts': 0,
                'career_wins': 0,
                'career_win_rate': 0,
                'career_place_rate': 0,
                'career_show_rate': 0,
                'avg_finish_position': 0,
                'avg_odds': 0,
                'days_since_last_race': 999,
            })
            return features
        
        # キャリア統計
        features['career_starts'] = len(past_results)
        features['career_wins'] = sum(1 for r in past_results if r.finish_position == 1)
        features['career_places'] = sum(1 for r in past_results if r.finish_position <= 2)
        features['career_shows'] = sum(1 for r in past_results if r.finish_position <= 3)
        
        features['career_win_rate'] = features['career_wins'] / features['career_starts']
        features['career_place_rate'] = features['career_places'] / features['career_starts']
        features['career_show_rate'] = features['career_shows'] / features['career_starts']
        
        # 平均着順
        features['avg_finish_position'] = np.mean([r.finish_position for r in past_results])
        
        # 平均オッズ
        odds_list = [r.odds for r in past_results if r.odds]
        features['avg_odds'] = np.mean(odds_list) if odds_list else 0
        
        # 前走からの間隔
        last_race = past_results[0]
        last_race_date = last_race.race.race_date
        features['days_since_last_race'] = (race_date - last_race_date).days
        
        # 直近N走の成績
        for n in [3, 5, 10]:
            recent_results = past_results[:n]
            if len(recent_results) >= n:
                features[f'last_{n}_avg_position'] = np.mean([r.finish_position for r in recent_results])
                features[f'last_{n}_win_rate'] = sum(1 for r in recent_results if r.finish_position == 1) / n
                features[f'last_{n}_show_rate'] = sum(1 for r in recent_results if r.finish_position <= 3) / n
        
        # スピード指数
        features.update(self._calculate_speed_figures(past_results))
        
        return features
    
    def _create_track_features(self, horse_id: str, race: Race) -> Dict:
        """距離・馬場適性特徴量を生成"""
        features = {}
        
        # 同条件での過去成績
        similar_results = self.db.query(RaceResult).join(Race).filter(
            RaceResult.horse_id == horse_id,
            Race.track == race.track,
            Race.race_date < race.race_date
        ).all()
        
        features['track_starts'] = len(similar_results)
        features['track_wins'] = sum(1 for r in similar_results if r.finish_position == 1)
        features['track_win_rate'] = features['track_wins'] / features['track_starts'] if features['track_starts'] > 0 else 0
        
        # 距離適性
        distance_results = self.db.query(RaceResult).join(Race).filter(
            RaceResult.horse_id == horse_id,
            Race.distance.between(race.distance - 200, race.distance + 200),
            Race.race_date < race.race_date
        ).all()
        
        features['distance_starts'] = len(distance_results)
        features['distance_wins'] = sum(1 for r in distance_results if r.finish_position == 1)
        features['distance_win_rate'] = features['distance_wins'] / features['distance_starts'] if features['distance_starts'] > 0 else 0
        
        # 馬場適性
        surface_results = self.db.query(RaceResult).join(Race).filter(
            RaceResult.horse_id == horse_id,
            Race.surface == race.surface,
            Race.race_date < race.race_date
        ).all()
        
        features['surface_starts'] = len(surface_results)
        features['surface_wins'] = sum(1 for r in surface_results if r.finish_position == 1)
        features['surface_win_rate'] = features['surface_wins'] / features['surface_starts'] if features['surface_starts'] > 0 else 0
        
        return features
    
    def _create_human_features(self, horse_id: str, race_date: date) -> Dict:
        """騎手・調教師特徴量を生成"""
        features = {}
        
        # 最新の騎手・調教師情報を取得
        recent_result = self.db.query(RaceResult).join(Race).filter(
            RaceResult.horse_id == horse_id,
            Race.race_date < race_date
        ).order_by(Race.race_date.desc()).first()
        
        if recent_result and recent_result.jockey_id:
            # 騎手の成績
            jockey_results = self.db.query(RaceResult).join(Race).filter(
                RaceResult.jockey_id == recent_result.jockey_id,
                Race.race_date >= race_date - timedelta(days=365),
                Race.race_date < race_date
            ).all()
            
            features['jockey_recent_starts'] = len(jockey_results)
            features['jockey_recent_wins'] = sum(1 for r in jockey_results if r.finish_position == 1)
            features['jockey_recent_win_rate'] = features['jockey_recent_wins'] / features['jockey_recent_starts'] if features['jockey_recent_starts'] > 0 else 0
            
            # 馬との相性
            horse_jockey_results = self.db.query(RaceResult).join(Race).filter(
                RaceResult.horse_id == horse_id,
                RaceResult.jockey_id == recent_result.jockey_id,
                Race.race_date < race_date
            ).all()
            
            features['horse_jockey_starts'] = len(horse_jockey_results)
            features['horse_jockey_wins'] = sum(1 for r in horse_jockey_results if r.finish_position == 1)
            features['horse_jockey_compatibility'] = features['horse_jockey_wins'] / features['horse_jockey_starts'] if features['horse_jockey_starts'] > 0 else 0
        
        return features
    
    def _create_pedigree_features(self, horse: Horse) -> Dict:
        """血統特徴量を生成"""
        features = {}
        
        # 父系の成績
        if horse.father_id:
            father_offspring = self.db.query(Horse).filter_by(father_id=horse.father_id).all()
            father_results = []
            for offspring in father_offspring:
                results = self.db.query(RaceResult).filter_by(horse_id=offspring.horse_id).all()
                father_results.extend(results)
            
            if father_results:
                features['father_offspring_win_rate'] = sum(1 for r in father_results if r.finish_position == 1) / len(father_results)
            else:
                features['father_offspring_win_rate'] = 0
        
        return features
    
    def _create_form_features(self, horse_id: str, race_date: date) -> Dict:
        """調子・フォーム特徴量を生成"""
        features = {}
        
        # 最近のパフォーマンストレンド
        recent_results = self.db.query(RaceResult).join(Race).filter(
            RaceResult.horse_id == horse_id,
            Race.race_date < race_date
        ).order_by(Race.race_date.desc()).limit(5).all()
        
        if len(recent_results) >= 2:
            positions = [r.finish_position for r in recent_results]
            # 順位の改善/悪化トレンド
            features['position_trend'] = np.polyfit(range(len(positions)), positions, 1)[0]
            
            # 調子の安定性
            features['form_stability'] = np.std(positions)
        
        return features
    
    def _calculate_speed_figures(self, results: List[RaceResult]) -> Dict:
        """スピード指数を計算"""
        features = {}
        speed_figures = []
        
        for result in results[:10]:  # 直近10走
            if result.time_seconds and result.race:
                # 基準タイム（仮定）
                base_time = result.race.distance / 1000 * 60  # 1000mあたり60秒
                
                # スピード指数 = (基準タイム - 実際のタイム) * 10 + 基準値
                speed_figure = (base_time - result.time_seconds) * 10 + 100
                
                # 馬場状態による補正
                if result.race.track_condition == '不良':
                    speed_figure -= 5
                elif result.race.track_condition == '重':
                    speed_figure -= 3
                elif result.race.track_condition == '稍重':
                    speed_figure -= 1
                
                speed_figures.append(speed_figure)
        
        if speed_figures:
            features['avg_speed_figure'] = np.mean(speed_figures)
            features['max_speed_figure'] = np.max(speed_figures)
            features['speed_figure_trend'] = np.polyfit(range(len(speed_figures)), speed_figures, 1)[0] if len(speed_figures) >= 2 else 0
        
        return features
    
    def _add_race_level_features(self, df: pd.DataFrame, race: Race) -> pd.DataFrame:
        """レース全体の特徴量を追加"""
        # 出走頭数
        df['field_size'] = len(df)
        
        # レースグレード
        grade_mapping = {'G1': 5, 'G2': 4, 'G3': 3, 'オープン': 2, 'その他': 1}
        df['race_grade_numeric'] = grade_mapping.get(race.grade, 1)
        
        # 相対的な特徴量
        for col in ['avg_speed_figure', 'career_win_rate', 'avg_odds']:
            if col in df.columns:
                df[f'{col}_relative'] = df[col] / df[col].mean() if df[col].mean() > 0 else 0
                df[f'{col}_rank'] = df[col].rank(ascending=False)
        
        return df
    
    def close(self):
        """リソースをクリーンアップ"""
        self.db.close()