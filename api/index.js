export default function handler(req, res) {
  console.log('=== API Request Debug Info ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('Path:', req.url.split('?')[0]);
  console.log('Query:', req.query);
  console.log('Headers:', req.headers);
  console.log('===========================');

  try {
    const { query, method } = req;
    const path = req.url.split('?')[0];

    if (method !== 'GET') {
      console.log('ERROR: Method not allowed:', method);
      res.status(405).json({ 
        error: 'Method not allowed',
        received_method: method,
        allowed_methods: ['GET']
      });
      return;
    }

    // APIステータス
    if (path === '/api' || path === '/api/') {
      console.log('Handling API status request');
      const response = {
        message: '競馬データ分析API',
        version: '1.0',
        status: 'active',
        debug_info: {
          request_path: path,
          request_method: method,
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
      return;
    }

    // テストエンドポイント
    if (path === '/api/test') {
      console.log('Handling test endpoint request');
      const response = {
        status: 'success',
        message: 'API is working correctly',
        timestamp: new Date().toISOString(),
        debug_info: {
          node_version: process.version,
          platform: process.platform,
          memory_usage: process.memoryUsage(),
          uptime: process.uptime()
        }
      };
      console.log('Sending response:', response);
      res.status(200).json(response);
      return;
    }

    // レース一覧
    if (path === '/api/races') {
      console.log('Handling races list request');
      const date = query.date || new Date().toISOString().slice(0, 10).replace(/-/g, '');
      console.log('Using date:', date);
      const response = {
        date: date,
        count: 2,
        debug_info: {
          received_query: query,
          used_date: date
        },
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
      };
      console.log('Sending response:', response);
      res.status(200).json(response);
      return;
    }

    // レース詳細
    if (path.startsWith('/api/race/')) {
      console.log('Handling race detail request');
      const raceId = path.split('/').pop();
      console.log('Race ID:', raceId);
      const response = {
        race_id: raceId,
        race_name: 'サンプルレース',
        race_date: new Date().toISOString().slice(0, 10),
        course: '東京',
        distance: 1600,
        track_type: '芝',
        track_condition: '良',
        weather: '晴',
        debug_info: {
          requested_path: path,
          extracted_race_id: raceId
        },
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
      };
      console.log('Sending response:', response);
      res.status(200).json(response);
      return;
    }

    // 404
    console.log('No matching endpoint for path:', path);
    res.status(404).json({ 
      error: 'Endpoint not found',
      requested_path: path,
      available_endpoints: ['/api', '/api/test', '/api/races', '/api/race/{race_id}']
    });

  } catch (error) {
    console.error('=== API Error ===');
    console.error('Error message:', error.message);
    console.error('Error stack:', error.stack);
    console.error('================');
    
    res.status(500).json({ 
      error: 'Internal server error',
      message: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
}