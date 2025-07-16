export default function handler(req, res) {
  console.log('=== /api/races Request ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('Query:', req.query);
  console.log('========================');

  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const date = req.query.date || new Date().toISOString().slice(0, 10).replace(/-/g, '');
  
  const response = {
    date: date,
    count: 2,
    debug_info: {
      received_query: req.query,
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
}