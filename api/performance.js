export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { days = 30 } = req.query;

  // モックパフォーマンスデータ
  const mockPerformance = {
    model_version: "1.0.0",
    period_days: parseInt(days),
    metrics: {
      accuracy: 0.72,
      roi: 0.12,
      sharp_ratio: 1.35,
      max_drawdown: 0.18,
      win_rate: 0.16,
      place_rate: 0.42
    },
    daily_performance: [
      {
        date: "2024-02-25",
        accuracy: 0.75,
        roi: 0.15,
        predictions: 12,
        wins: 2,
        total_bets: 120000,
        total_returns: 138000
      },
      {
        date: "2024-02-24",
        accuracy: 0.68,
        roi: 0.08,
        predictions: 15,
        wins: 2,
        total_bets: 150000,
        total_returns: 162000
      },
      {
        date: "2024-02-23",
        accuracy: 0.73,
        roi: 0.11,
        predictions: 10,
        wins: 1,
        total_bets: 100000,
        total_returns: 111000
      }
    ]
  };

  res.status(200).json(mockPerformance);
}