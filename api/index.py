from http.server import BaseHTTPRequestHandler
from datetime import datetime
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        
        # APIステータス
        if path == '/api' or path == '/api/':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'message': '競馬データ分析API',
                'version': '1.0',
                'status': 'active',
                'endpoints': {
                    '/api': 'APIステータス',
                    '/api/test': 'テストエンドポイント',
                    '/api/races': 'レース一覧を取得（開発中）',
                    '/api/race/{race_id}': 'レース詳細情報を取得（開発中）'
                }
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # テストエンドポイント
        elif path == '/api/test':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {
                'status': 'success',
                'message': 'API is working correctly',
                'timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # レース一覧
        elif path == '/api/races':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            date_str = datetime.now().strftime('%Y%m%d')
            response = {
                'date': date_str,
                'count': 2,
                'races': [
                    {
                        'race_id': 'R001',
                        'race_name': 'サンプルレース1',
                        'race_number': 1,
                        'course': '東京',
                        'distance': 1600,
                        'track_type': '芝',
                        'start_time': '10:00'
                    },
                    {
                        'race_id': 'R002',
                        'race_name': 'サンプルレース2',
                        'race_number': 2,
                        'course': '東京',
                        'distance': 2000,
                        'track_type': 'ダート',
                        'start_time': '10:30'
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # レース詳細
        elif path.startswith('/api/race/'):
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            race_id = path.split('/')[-1]
            response = {
                'race_id': race_id,
                'race_name': 'サンプルレース',
                'race_date': datetime.now().strftime('%Y-%m-%d'),
                'course': '東京',
                'distance': 1600,
                'track_type': '芝',
                'track_condition': '良',
                'weather': '晴',
                'results': [
                    {
                        'ranking': 1,
                        'horse_number': 1,
                        'horse_name': 'サンプルホース1',
                        'jockey': 'サンプル騎手1',
                        'time': '1:33.5',
                        'odds': 2.1,
                        'popularity': 1
                    }
                ]
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        # 404
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'error': 'Endpoint not found'}
            self.wfile.write(json.dumps(response).encode())
            return