export default function handler(req, res) {
  console.log('=== Prediction Request ===');
  console.log('Method:', req.method);
  console.log('Query:', req.query);
  console.log('=========================');

  if (req.method !== 'GET' && req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const raceId = req.query.race_id || 'R001';
  
  // 実際の予測システムのデモレスポンス
  const predictions = [
    {
      horse_id: '001',
      horse_name: 'ディープインパクト',
      post_position: 1,
      predictions: {
        win_probability: 0.285,
        place_probability: 0.652,
        model_confidence: 0.89
      },
      features: {
        normalized_speed_index: 115.2,
        class_rating: 0.95,
        recent_form: {
          last_3_races: [1, 2, 1],
          avg_speed_index: 113.5
        },
        jockey_horse_synergy: 0.85
      },
      expected_value: {
        win: 1.425,  // 勝率28.5% × オッズ5.0
        place: 1.108  // 複勝率65.2% × オッズ1.7
      }
    },
    {
      horse_id: '002',
      horse_name: 'オルフェーヴル',
      post_position: 2,
      predictions: {
        win_probability: 0.198,
        place_probability: 0.523,
        model_confidence: 0.82
      },
      features: {
        normalized_speed_index: 112.1,
        class_rating: 0.88,
        recent_form: {
          last_3_races: [2, 3, 1],
          avg_speed_index: 110.8
        },
        jockey_horse_synergy: 0.72
      },
      expected_value: {
        win: 1.386,  // 勝率19.8% × オッズ7.0
        place: 1.046  // 複勝率52.3% × オッズ2.0
      }
    },
    {
      horse_id: '003',
      horse_name: 'キタサンブラック',
      post_position: 3,
      predictions: {
        win_probability: 0.165,
        place_probability: 0.412,
        model_confidence: 0.75
      },
      features: {
        normalized_speed_index: 109.8,
        class_rating: 0.82,
        recent_form: {
          last_3_races: [3, 1, 4],
          avg_speed_index: 108.3
        },
        jockey_horse_synergy: 0.68
      },
      expected_value: {
        win: 1.485,  // 勝率16.5% × オッズ9.0
        place: 0.988  // 複勝率41.2% × オッズ2.4
      }
    }
  ];

  // 予測結果を勝率順にソート
  predictions.sort((a, b) => b.predictions.win_probability - a.predictions.win_probability);

  const response = {
    race_id: raceId,
    race_info: {
      course: '東京競馬場',
      distance: 2000,
      track_type: '芝',
      track_condition: '良',
      weather: '晴',
      temperature: 22.5,
      humidity: 45
    },
    model_metadata: {
      version: '2.0',
      last_updated: '2025-07-16T10:00:00Z',
      training_races: 50000,
      validation_accuracy: 0.752
    },
    predictions: predictions,
    recommended_bets: predictions
      .filter(p => p.expected_value.win > 1.05 || p.expected_value.place > 1.05)
      .map(p => ({
        horse_id: p.horse_id,
        horse_name: p.horse_name,
        bet_type: p.expected_value.win > p.expected_value.place ? 'win' : 'place',
        expected_value: Math.max(p.expected_value.win, p.expected_value.place),
        confidence: p.predictions.model_confidence
      })),
    analysis_timestamp: new Date().toISOString()
  };

  console.log('Sending prediction response');
  res.status(200).json(response);
}