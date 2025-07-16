export default function handler(req, res) {
  console.log('=== /api Request ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('====================');

  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const response = {
    message: '競馬データ分析API',
    version: '1.0',
    status: 'active',
    debug_info: {
      request_path: req.url,
      request_method: req.method,
      timestamp: new Date().toISOString()
    },
    endpoints: {
      '/api': 'APIステータス',
      '/api/test': 'テストエンドポイント',
      '/api/races': 'レース一覧を取得（開発中）',
      '/api/race/{race_id}': 'レース詳細情報を取得（開発中）'
    }
  };

  console.log('Sending response:', response);
  res.status(200).json(response);
}