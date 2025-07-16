export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { date = new Date().toISOString().split('T')[0] } = req.query;

  // モックレースリストデータ
  const mockRaces = {
    count: 3,
    races: [
      {
        race_id: "2024022503010311",
        race_name: "第3回中山記念",
        race_date: date,
        track: "中山",
        distance: 1800,
        surface: "芝",
        grade: "G2",
        start_time: "15:45",
        entries_count: 12
      },
      {
        race_id: "2024022504021012",
        race_name: "阪急杯",
        race_date: date,
        track: "阪神",
        distance: 1400,
        surface: "芝",
        grade: "G3",
        start_time: "15:35",
        entries_count: 18
      },
      {
        race_id: "2024022505031211",
        race_name: "小倉大賞典",
        race_date: date,
        track: "小倉",
        distance: 1800,
        surface: "芝",
        grade: "G3",
        start_time: "15:20",
        entries_count: 14
      }
    ]
  };

  res.status(200).json(mockRaces);
}