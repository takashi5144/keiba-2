export default function handler(req, res) {
  console.log('=== /api/race/[id] Request ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('Query:', req.query);
  console.log('Race ID:', req.query.id);
  console.log('==============================');

  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const raceId = req.query.id || 'R001';
  
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
      requested_path: req.url,
      extracted_race_id: raceId,
      query_params: req.query
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
}