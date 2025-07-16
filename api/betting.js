export default function handler(req, res) {
  console.log('=== Betting Calculation Request ===');
  console.log('Method:', req.method);
  console.log('Query:', req.query);
  console.log('==================================');

  if (req.method !== 'GET' && req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  // リクエストパラメータ
  const bankroll = parseFloat(req.query.bankroll || '100000');
  const raceId = req.query.race_id || 'R001';
  const riskLevel = req.query.risk_level || 'moderate';

  // リスクレベルに応じたケリー係数
  const kellyMultipliers = {
    conservative: 0.15,
    moderate: 0.25,
    aggressive: 0.40
  };

  const kellyFraction = kellyMultipliers[riskLevel] || 0.25;

  // 賭けの機会を計算
  const bettingOpportunities = [
    {
      horse_id: '001',
      horse_name: 'ディープインパクト',
      bet_type: 'win',
      market_odds: 5.0,
      predicted_probability: 0.285,
      expected_value: 1.425,
      kelly_calculation: {
        full_kelly: 0.106,  // (4 * 0.285 - 0.715) / 4 = 0.106
        fractional_kelly: 0.0265,  // 0.106 * 0.25
        recommended_percentage: 0.025  // 最大2.5%制限
      },
      recommended_bet: Math.min(bankroll * 0.0265, bankroll * 0.025),
      confidence: 0.89,
      risk_assessment: 'moderate'
    },
    {
      horse_id: '003',
      horse_name: 'キタサンブラック',
      bet_type: 'win',
      market_odds: 9.0,
      predicted_probability: 0.165,
      expected_value: 1.485,
      kelly_calculation: {
        full_kelly: 0.123,  // (8 * 0.165 - 0.835) / 8 = 0.123
        fractional_kelly: 0.0308,  // 0.123 * 0.25
        recommended_percentage: 0.025  // 最大2.5%制限
      },
      recommended_bet: Math.min(bankroll * 0.0308, bankroll * 0.025),
      confidence: 0.75,
      risk_assessment: 'high'
    }
  ];

  // 期待値が1.05以上のものだけフィルタ
  const validBets = bettingOpportunities.filter(bet => bet.expected_value >= 1.05);
  
  // 合計賭け金を計算
  const totalBetAmount = validBets.reduce((sum, bet) => sum + bet.recommended_bet, 0);
  
  // 期待リターンを計算
  const expectedReturn = validBets.reduce((sum, bet) => 
    sum + (bet.recommended_bet * (bet.expected_value - 1)), 0
  );

  const response = {
    race_id: raceId,
    bankroll_info: {
      current_bankroll: bankroll,
      risk_level: riskLevel,
      kelly_fraction: kellyFraction,
      max_bet_percentage: 0.025
    },
    betting_opportunities: validBets,
    summary: {
      total_recommended_bets: validBets.length,
      total_bet_amount: totalBetAmount,
      expected_return: expectedReturn,
      expected_roi: (expectedReturn / totalBetAmount * 100).toFixed(2) + '%',
      bankroll_utilization: (totalBetAmount / bankroll * 100).toFixed(2) + '%'
    },
    risk_metrics: {
      max_drawdown_limit: '20%',
      current_exposure: (totalBetAmount / bankroll * 100).toFixed(2) + '%',
      diversification_score: validBets.length > 1 ? 'good' : 'poor',
      confidence_weighted_ev: validBets.reduce((sum, bet) => 
        sum + (bet.expected_value * bet.confidence), 0
      ) / validBets.length
    },
    warnings: totalBetAmount > bankroll * 0.1 ? 
      ['賭け金が資金の10%を超えています'] : [],
    timestamp: new Date().toISOString()
  };

  console.log('Sending betting calculation response');
  res.status(200).json(response);
}