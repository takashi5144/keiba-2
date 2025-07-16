from datetime import datetime

def handler(request):
    """Vercel用のハンドラー関数"""
    path = request.url.split('?')[0].split('/')[-1] or ''
    
    # APIステータス
    if path == 'api' or path == '':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': {
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
        }
    
    # テストエンドポイント
    elif path == 'test':
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': {
                'status': 'success',
                'message': 'API is working correctly',
                'timestamp': datetime.now().isoformat()
            }
        }
    
    # レース一覧
    elif path == 'races':
        date_str = request.url.split('date=')[-1] if 'date=' in request.url else datetime.now().strftime('%Y%m%d')
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': {
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
        }
    
    # レース詳細
    elif path.startswith('race'):
        race_id = path.split('/')[-1] if '/' in path else 'R001'
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': {
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
        }
    
    # 404
    else:
        return {
            'statusCode': 404,
            'headers': {'Content-Type': 'application/json'},
            'body': {'error': 'Endpoint not found'}
        }