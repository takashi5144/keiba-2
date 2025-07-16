export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // レースIDを取得
  const { raceId } = req.query;

  if (!raceId) {
    res.status(400).json({ error: 'レースIDが必要です' });
    return;
  }

  // モックデータを返す（本番環境では実際のPython APIにプロキシする）
  const mockPredictions = {
    race_id: raceId,
    race_info: {
      race_id: raceId,
      race_name: "第3回中山記念",
      race_date: "2024-02-25",
      track: "中山",
      distance: 1800,
      surface: "芝",
      grade: "G2"
    },
    model_version: "1.0.0",
    predictions: [
      {
        horse_id: "H001",
        horse_name: "イクイノックス",
        post_position: 1,
        win_probability: 0.35,
        place_probability: 0.65,
        show_probability: 0.85,
        confidence_score: 0.92,
        ensemble_score: 0.88,
        expected_value: 2.1
      },
      {
        horse_id: "H002",
        horse_name: "ジャスティンパレス",
        post_position: 2,
        win_probability: 0.22,
        place_probability: 0.48,
        show_probability: 0.72,
        confidence_score: 0.85,
        ensemble_score: 0.76,
        expected_value: 1.8
      },
      {
        horse_id: "H003",
        horse_name: "ダノンザキッド",
        post_position: 3,
        win_probability: 0.18,
        place_probability: 0.42,
        show_probability: 0.68,
        confidence_score: 0.82,
        ensemble_score: 0.71,
        expected_value: 1.6
      }
    ],
    timestamp: new Date().toISOString()
  };

  res.status(200).json(mockPredictions);
}