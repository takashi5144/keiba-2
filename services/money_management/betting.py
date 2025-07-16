"""
資金管理と賭け戦略
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from decimal import Decimal
from datetime import datetime
from loguru import logger

from database.models.prediction import Prediction, BettingRecommendation
from database.models.base import SessionLocal

class BettingConfig:
    """賭け設定"""
    KELLY_FRACTIONS = {
        "conservative": 0.15,
        "moderate": 0.25,
        "aggressive": 0.40
    }
    
    MAX_BET_PERCENTAGES = {
        "conservative": 0.015,  # 1.5%
        "moderate": 0.025,      # 2.5%
        "aggressive": 0.04      # 4%
    }
    
    MIN_EXPECTED_VALUES = {
        "conservative": 1.10,   # 10%以上の期待値
        "moderate": 1.05,       # 5%以上の期待値
        "aggressive": 1.02      # 2%以上の期待値
    }
    
    BET_TYPES = {
        "win": {"min_odds": 1.5, "max_horses": 1},
        "place": {"min_odds": 1.2, "max_horses": 1},
        "exacta": {"min_odds": 10.0, "max_horses": 2},
        "quinella": {"min_odds": 5.0, "max_horses": 2},
        "trifecta": {"min_odds": 50.0, "max_horses": 3}
    }

class BettingStrategy:
    """賭け戦略クラス"""
    
    def __init__(self):
        self.config = BettingConfig()
        self.db = SessionLocal()
    
    def calculate_optimal_bets(self, predictions: List[Prediction], 
                             bankroll: float, risk_level: str = "moderate") -> Dict:
        """最適な賭けを計算"""
        logger.info(f"Calculating optimal bets for {len(predictions)} horses")
        
        # リスクレベルの設定を取得
        kelly_fraction = self.config.KELLY_FRACTIONS.get(risk_level, 0.25)
        max_bet_percentage = self.config.MAX_BET_PERCENTAGES.get(risk_level, 0.025)
        min_expected_value = self.config.MIN_EXPECTED_VALUES.get(risk_level, 1.05)
        
        recommendations = []
        
        # 各馬券種別に評価
        for bet_type, bet_config in self.config.BET_TYPES.items():
            if bet_type == "win":
                win_bets = self._evaluate_win_bets(
                    predictions, bankroll, kelly_fraction, 
                    max_bet_percentage, min_expected_value
                )
                recommendations.extend(win_bets)
            
            elif bet_type == "place":
                place_bets = self._evaluate_place_bets(
                    predictions, bankroll, kelly_fraction,
                    max_bet_percentage, min_expected_value
                )
                recommendations.extend(place_bets)
            
            # TODO: exacta, quinella, trifectaの実装
        
        # 推奨を期待値でソート
        recommendations.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # 資金配分を最適化
        optimized_recommendations = self._optimize_portfolio(
            recommendations, bankroll, max_bet_percentage
        )
        
        # サマリーを計算
        summary = self._calculate_summary(optimized_recommendations, bankroll)
        
        # 警告を生成
        warnings = self._generate_warnings(optimized_recommendations, bankroll, risk_level)
        
        return {
            "race_id": predictions[0].race_id if predictions else None,
            "bankroll": bankroll,
            "risk_level": risk_level,
            "recommendations": optimized_recommendations,
            "summary": summary,
            "warnings": warnings,
            "timestamp": datetime.now()
        }
    
    def _evaluate_win_bets(self, predictions: List[Prediction], bankroll: float,
                          kelly_fraction: float, max_bet_percentage: float,
                          min_expected_value: float) -> List[Dict]:
        """単勝の評価"""
        recommendations = []
        
        for pred in predictions:
            # オッズを取得（実際はデータベースから取得）
            odds = self._get_current_odds(pred.horse_id, "win")
            if not odds or odds < 1.5:  # 最低オッズ
                continue
            
            # 期待値を計算
            expected_value = float(pred.win_probability) * odds
            
            if expected_value < min_expected_value:
                continue
            
            # ケリー基準で賭け金を計算
            kelly = self._calculate_kelly_fraction(
                float(pred.win_probability), odds
            )
            
            if kelly <= 0:
                continue
            
            # フラクショナル・ケリーを適用
            adjusted_kelly = kelly * kelly_fraction
            
            # 最大賭け金制限
            bet_percentage = min(adjusted_kelly, max_bet_percentage)
            bet_amount = bankroll * bet_percentage
            
            # 期待リターンを計算
            expected_return = bet_amount * (expected_value - 1)
            
            recommendations.append({
                "bet_type": "win",
                "selections": [pred.horse_id],
                "probability": float(pred.win_probability),
                "odds": odds,
                "kelly_fraction": kelly,
                "recommended_percentage": bet_percentage,
                "recommended_amount": bet_amount,
                "expected_value": expected_value,
                "expected_return": expected_return,
                "confidence_score": float(pred.confidence_score),
                "risk_level": self._assess_risk_level(expected_value, kelly)
            })
        
        return recommendations
    
    def _evaluate_place_bets(self, predictions: List[Prediction], bankroll: float,
                           kelly_fraction: float, max_bet_percentage: float,
                           min_expected_value: float) -> List[Dict]:
        """複勝の評価"""
        recommendations = []
        
        for pred in predictions:
            # 複勝オッズを取得
            odds = self._get_current_odds(pred.horse_id, "place")
            if not odds or odds < 1.2:
                continue
            
            # 期待値を計算
            expected_value = float(pred.place_probability) * odds
            
            if expected_value < min_expected_value:
                continue
            
            # ケリー基準で賭け金を計算
            kelly = self._calculate_kelly_fraction(
                float(pred.place_probability), odds
            )
            
            if kelly <= 0:
                continue
            
            # フラクショナル・ケリーを適用
            adjusted_kelly = kelly * kelly_fraction
            
            # 最大賭け金制限
            bet_percentage = min(adjusted_kelly, max_bet_percentage)
            bet_amount = bankroll * bet_percentage
            
            # 期待リターンを計算
            expected_return = bet_amount * (expected_value - 1)
            
            recommendations.append({
                "bet_type": "place",
                "selections": [pred.horse_id],
                "probability": float(pred.place_probability),
                "odds": odds,
                "kelly_fraction": kelly,
                "recommended_percentage": bet_percentage,
                "recommended_amount": bet_amount,
                "expected_value": expected_value,
                "expected_return": expected_return,
                "confidence_score": float(pred.confidence_score),
                "risk_level": self._assess_risk_level(expected_value, kelly)
            })
        
        return recommendations
    
    def _calculate_kelly_fraction(self, probability: float, odds: float) -> float:
        """ケリー基準による最適賭け金割合を計算"""
        if odds <= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        return max(0, kelly)
    
    def _optimize_portfolio(self, recommendations: List[Dict], 
                          bankroll: float, max_bet_percentage: float) -> List[Dict]:
        """ポートフォリオ最適化"""
        if not recommendations:
            return []
        
        # 総賭け金が資金の一定割合を超えないように調整
        total_bet_amount = sum(rec['recommended_amount'] for rec in recommendations)
        max_total_bet = bankroll * max_bet_percentage * 3  # 3つまでの賭け
        
        if total_bet_amount > max_total_bet:
            # 比例配分で調整
            scale_factor = max_total_bet / total_bet_amount
            for rec in recommendations:
                rec['recommended_amount'] *= scale_factor
                rec['recommended_percentage'] *= scale_factor
                rec['expected_return'] *= scale_factor
        
        # 上位N個の推奨のみを返す
        return recommendations[:5]
    
    def _calculate_summary(self, recommendations: List[Dict], bankroll: float) -> Dict:
        """サマリーを計算"""
        if not recommendations:
            return {
                "total_bets": 0,
                "total_amount": 0,
                "expected_return": 0,
                "expected_roi": 0,
                "average_confidence": 0,
                "risk_distribution": {}
            }
        
        total_amount = sum(rec['recommended_amount'] for rec in recommendations)
        expected_return = sum(rec['expected_return'] for rec in recommendations)
        
        risk_distribution = {}
        for rec in recommendations:
            risk_level = rec['risk_level']
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        return {
            "total_bets": len(recommendations),
            "total_amount": total_amount,
            "expected_return": expected_return,
            "expected_roi": (expected_return / total_amount * 100) if total_amount > 0 else 0,
            "average_confidence": np.mean([rec['confidence_score'] for rec in recommendations]),
            "risk_distribution": risk_distribution
        }
    
    def _generate_warnings(self, recommendations: List[Dict], 
                         bankroll: float, risk_level: str) -> List[str]:
        """警告を生成"""
        warnings = []
        
        total_amount = sum(rec['recommended_amount'] for rec in recommendations)
        
        # 総賭け金が資金の10%を超える
        if total_amount > bankroll * 0.1:
            warnings.append(f"総賭け金が資金の{total_amount/bankroll*100:.1f}%に達しています")
        
        # 高リスクの賭けが多い
        high_risk_count = sum(1 for rec in recommendations if rec['risk_level'] == 'high')
        if high_risk_count >= 3:
            warnings.append(f"{high_risk_count}個の高リスク賭けが含まれています")
        
        # 信頼度が低い予測
        low_confidence = [rec for rec in recommendations if rec['confidence_score'] < 0.6]
        if low_confidence:
            warnings.append(f"{len(low_confidence)}個の低信頼度予測が含まれています")
        
        return warnings
    
    def _get_current_odds(self, horse_id: str, bet_type: str) -> Optional[float]:
        """現在のオッズを取得（仮実装）"""
        # 実際はリアルタイムオッズAPIから取得
        import random
        
        if bet_type == "win":
            return random.uniform(1.5, 50.0)
        elif bet_type == "place":
            return random.uniform(1.1, 5.0)
        else:
            return None
    
    def _assess_risk_level(self, expected_value: float, kelly_fraction: float) -> str:
        """リスクレベルを評価"""
        if expected_value > 1.5 and kelly_fraction > 0.1:
            return "low"
        elif expected_value > 1.2 or kelly_fraction > 0.05:
            return "medium"
        else:
            return "high"
    
    def save_recommendations(self, race_id: str, recommendations: List[Dict]):
        """推奨をデータベースに保存"""
        for rec in recommendations:
            betting_rec = BettingRecommendation(
                race_id=race_id,
                bet_type=rec['bet_type'],
                selections={"horses": rec['selections']},
                kelly_fraction=rec['kelly_fraction'],
                recommended_percentage=rec['recommended_percentage'],
                expected_value=rec['expected_value'],
                expected_return=rec['expected_return'],
                risk_level=rec['risk_level'],
                confidence_score=rec['confidence_score']
            )
            self.db.add(betting_rec)
        
        self.db.commit()
    
    def close(self):
        """リソースをクリーンアップ"""
        self.db.close()