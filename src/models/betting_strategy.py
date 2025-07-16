"""
期待値計算と資金管理戦略
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class BettingOpportunity:
    """賭けの機会を表すデータクラス"""
    race_id: str
    horse_id: str
    bet_type: str
    predicted_probability: float
    market_odds: float
    expected_value: float
    kelly_fraction: float
    recommended_bet_size: float
    confidence: float

class KellyBettingStrategy:
    """ケリー基準に基づく賭け戦略"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.kelly_fraction = config['kelly_fraction']
        self.max_bet_percentage = config['max_bet_percentage']
        self.min_expected_value = config['min_expected_value']
        self.confidence_threshold = config['confidence_threshold']
        
    def calculate_expected_value(self, predicted_prob: float, 
                               market_odds: float) -> float:
        """期待値を計算"""
        return predicted_prob * market_odds
    
    def calculate_kelly_fraction(self, predicted_prob: float,
                               market_odds: float) -> float:
        """ケリー基準による最適賭け金割合を計算"""
        if market_odds <= 1:
            return 0.0
            
        # ケリー公式: f = (bp - q) / b
        # b = オッズ - 1, p = 勝率, q = 1 - p
        b = market_odds - 1
        p = predicted_prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        # 負の値は賭けない
        return max(0, kelly)
    
    def apply_fractional_kelly(self, kelly_fraction: float) -> float:
        """フラクショナル・ケリーを適用"""
        return kelly_fraction * self.kelly_fraction
    
    def calculate_bet_size(self, bankroll: float, kelly_fraction: float,
                         confidence: float) -> float:
        """実際の賭け金サイズを計算"""
        # フラクショナル・ケリーを適用
        adjusted_kelly = self.apply_fractional_kelly(kelly_fraction)
        
        # 信頼度による調整
        confidence_adjusted = adjusted_kelly * confidence
        
        # 最大賭け金制限
        max_bet = bankroll * self.max_bet_percentage
        
        # 推奨賭け金
        recommended_bet = bankroll * confidence_adjusted
        
        return min(recommended_bet, max_bet)
    
    def evaluate_betting_opportunities(self, predictions: pd.DataFrame,
                                     market_odds: pd.DataFrame,
                                     bankroll: float) -> List[BettingOpportunity]:
        """賭けの機会を評価"""
        opportunities = []
        
        for idx, row in predictions.iterrows():
            # 各馬券種別に評価
            for bet_type in ['win', 'place', 'exacta', 'trifecta']:
                if bet_type not in market_odds.columns:
                    continue
                    
                predicted_prob = row[f'{bet_type}_probability']
                odds = market_odds.loc[idx, bet_type]
                confidence = row[f'{bet_type}_confidence']
                
                # 期待値計算
                ev = self.calculate_expected_value(predicted_prob, odds)
                
                # 期待値と信頼度の閾値チェック
                if ev < self.min_expected_value or confidence < self.confidence_threshold:
                    continue
                
                # ケリー基準による賭け金計算
                kelly = self.calculate_kelly_fraction(predicted_prob, odds)
                bet_size = self.calculate_bet_size(bankroll, kelly, confidence)
                
                # 賭け金が0より大きい場合のみ追加
                if bet_size > 0:
                    opportunities.append(BettingOpportunity(
                        race_id=row['race_id'],
                        horse_id=row['horse_id'],
                        bet_type=bet_type,
                        predicted_probability=predicted_prob,
                        market_odds=odds,
                        expected_value=ev,
                        kelly_fraction=kelly,
                        recommended_bet_size=bet_size,
                        confidence=confidence
                    ))
        
        # 期待値の高い順にソート
        opportunities.sort(key=lambda x: x.expected_value, reverse=True)
        
        return opportunities
    
    def optimize_bet_allocation(self, opportunities: List[BettingOpportunity],
                              bankroll: float) -> Dict:
        """複数の賭けに対して最適な資金配分を計算"""
        if not opportunities:
            return {'bets': [], 'total_amount': 0, 'expected_return': 0}
        
        # 賭け金の合計が資金を超えないように調整
        total_recommended = sum(opp.recommended_bet_size for opp in opportunities)
        
        if total_recommended > bankroll:
            # 比例配分で調整
            scale_factor = bankroll / total_recommended
            for opp in opportunities:
                opp.recommended_bet_size *= scale_factor
        
        # 期待リターンを計算
        expected_return = sum(
            opp.recommended_bet_size * (opp.expected_value - 1)
            for opp in opportunities
        )
        
        return {
            'bets': opportunities,
            'total_amount': sum(opp.recommended_bet_size for opp in opportunities),
            'expected_return': expected_return,
            'expected_roi': expected_return / bankroll if bankroll > 0 else 0
        }

class RiskManagement:
    """リスク管理システム"""
    
    def __init__(self, max_drawdown_limit: float = 0.20):
        self.max_drawdown_limit = max_drawdown_limit
        self.peak_bankroll = 0
        self.current_drawdown = 0
        
    def update_bankroll(self, current_bankroll: float):
        """資金残高を更新してドローダウンを計算"""
        if current_bankroll > self.peak_bankroll:
            self.peak_bankroll = current_bankroll
            self.current_drawdown = 0
        else:
            self.current_drawdown = (self.peak_bankroll - current_bankroll) / self.peak_bankroll
    
    def should_stop_betting(self) -> bool:
        """ドローダウン制限に達したかチェック"""
        return self.current_drawdown >= self.max_drawdown_limit
    
    def calculate_sharpe_ratio(self, returns: List[float], 
                             risk_free_rate: float = 0.02) -> float:
        """シャープレシオを計算"""
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # 日次換算
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def calculate_metrics(self, betting_history: pd.DataFrame) -> Dict:
        """パフォーマンス指標を計算"""
        total_bets = len(betting_history)
        winning_bets = len(betting_history[betting_history['profit'] > 0])
        
        return {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'win_rate': winning_bets / total_bets if total_bets > 0 else 0,
            'total_profit': betting_history['profit'].sum(),
            'roi': betting_history['profit'].sum() / betting_history['bet_amount'].sum(),
            'average_odds': betting_history['odds'].mean(),
            'max_drawdown': self.current_drawdown,
            'sharpe_ratio': self.calculate_sharpe_ratio(betting_history['return'].tolist())
        }