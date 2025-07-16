export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { raceId, bankroll = 100000, riskLevel = 'moderate' } = req.query;

  if (!raceId) {
    res.status(400).json({ error: 'レースIDが必要です' });
    return;
  }

  // モック賭け推奨データ
  const mockBettingRecommendations = {
    race_id: raceId,
    bankroll: parseFloat(bankroll),
    risk_level: riskLevel,
    recommendations: [
      {
        bet_type: "win",
        selections: ["H001"],
        kelly_fraction: 0.15,
        recommended_percentage: 0.025,
        recommended_amount: parseFloat(bankroll) * 0.025,
        expected_value: 2.1,
        expected_return: parseFloat(bankroll) * 0.025 * 1.1,
        confidence_score: 0.92,
        risk_level: "low"
      },
      {
        bet_type: "place",
        selections: ["H002"],
        kelly_fraction: 0.12,
        recommended_percentage: 0.015,
        recommended_amount: parseFloat(bankroll) * 0.015,
        expected_value: 1.4,
        expected_return: parseFloat(bankroll) * 0.015 * 0.4,
        confidence_score: 0.85,
        risk_level: "medium"
      }
    ],
    summary: {
      total_bets: 2,
      total_amount: parseFloat(bankroll) * 0.04,
      expected_return: parseFloat(bankroll) * 0.04 * 0.15,
      expected_roi: 15.0,
      average_confidence: 0.885,
      risk_distribution: {
        low: 1,
        medium: 1,
        high: 0
      }
    },
    warnings: [],
    timestamp: new Date().toISOString()
  };

  res.status(200).json(mockBettingRecommendations);
}