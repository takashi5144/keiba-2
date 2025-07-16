export default function handler(req, res) {
  const { query, method } = req;
  const path = req.url.split('?')[0];

  if (method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  // APIステータス
  if (path === '/api' || path === '/api/') {
    res.status(200).json({
      message: '競馬データ分析API',
      version: '1.0',
      status: 'active',
      endpoints: {
        '/api': 'APIステータス',
        '/api/test': 'テストエンドポイント',
        '/api/races': 'レース一覧を取得（開発中）',
        '/api/race/{race_id}': 'レース詳細情報を取得（開発中）'
      }
    });
    return;
  }

  // テストエンドポイント
  if (path === '/api/test') {
    res.status(200).json({
      status: 'success',
      message: 'API is working correctly',
      timestamp: new Date().toISOString()
    });
    return;
  }

  // レース一覧
  if (path === '/api/races') {
    const date = query.date || new Date().toISOString().slice(0, 10).replace(/-/g, '');
    res.status(200).json({
      date: date,
      count: 2,
      races: [
        {
          race_id: 'R001',
          race_name: 'サンプルレース1',
          race_number: 1,
          course: '東京',
          distance: 1600,
          track_type: '芝',
          start_time: '10:00'
        },
        {
          race_id: 'R002',
          race_name: 'サンプルレース2',
          race_number: 2,
          course: '東京',
          distance: 2000,
          track_type: 'ダート',
          start_time: '10:30'
        }
      ]
    });
    return;
  }

  // レース詳細
  if (path.startsWith('/api/race/')) {
    const raceId = path.split('/').pop();
    res.status(200).json({
      race_id: raceId,
      race_name: 'サンプルレース',
      race_date: new Date().toISOString().slice(0, 10),
      course: '東京',
      distance: 1600,
      track_type: '芝',
      track_condition: '良',
      weather: '晴',
      results: [
        {
          ranking: 1,
          horse_number: 1,
          horse_name: 'サンプルホース1',
          jockey: 'サンプル騎手1',
          time: '1:33.5',
          odds: 2.1,
          popularity: 1
        }
      ]
    });
    return;
  }

  // 404
  res.status(404).json({ error: 'Endpoint not found' });
}