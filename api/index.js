export default function handler(req, res) {
  console.log('=== Horse Racing Prediction API ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('================================');

  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const response = {
    message: '高精度競馬予測システム API',
    version: '2.0',
    status: 'active',
    model_info: {
      ensemble: {
        lightgbm_weight: 0.7,
        neural_network_weight: 0.3
      },
      accuracy_metrics: {
        roi: '32.75%',
        sharpe_ratio: 1.45,
        win_rate: '15.2%',
        confidence_level: '95%'
      }
    },
    endpoints: {
      '/api': 'APIステータスと概要',
      '/api/predict': '単一レースの予測',
      '/api/analyze': 'レース分析と特徴量',
      '/api/betting': '最適賭け金計算',
      '/api/performance': 'モデルパフォーマンス'
    },
    features: {
      machine_learning: [
        'LightGBMアンサンブル',
        'ディープラーニング',
        'XGBoost（補助）'
      ],
      data_sources: [
        'レース履歴データ',
        'リアルタイムオッズ',
        '馬場状態',
        'セクショナルタイム'
      ],
      betting_strategy: [
        'ケリー基準',
        '期待値計算',
        'リスク管理'
      ]
    },
    timestamp: new Date().toISOString()
  };

  console.log('Sending response:', response);
  res.status(200).json(response);
}