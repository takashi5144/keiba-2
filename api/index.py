from flask import Flask, jsonify, request
from datetime import datetime
import os

app = Flask(__name__)

@app.route('/api', methods=['GET'])
def home():
    """APIのホームページ"""
    return jsonify({
        'message': '競馬データ分析API',
        'version': '1.0',
        'status': 'active',
        'endpoints': {
            '/api': 'APIステータス',
            '/api/test': 'テストエンドポイント',
            '/api/races': 'レース一覧を取得（開発中）',
            '/api/race/<race_id>': 'レース詳細情報を取得（開発中）'
        }
    })

@app.route('/api/test', methods=['GET'])
def test():
    """テストエンドポイント"""
    return jsonify({
        'status': 'success',
        'message': 'API is working correctly',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/races', methods=['GET'])
def get_races():
    """レース一覧を取得（モックデータ）"""
    date_str = request.args.get('date', datetime.now().strftime('%Y%m%d'))
    
    # モックデータを返す
    return jsonify({
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
    })

@app.route('/api/race/<race_id>', methods=['GET'])
def get_race_detail(race_id):
    """レース詳細情報を取得（モックデータ）"""
    return jsonify({
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
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Vercel用のエントリーポイント
def handler(request):
    """Vercel用のハンドラー関数"""
    with app.test_request_context(
        request.url,
        method=request.method,
        headers=request.headers,
        data=request.get_data()
    ):
        response = app.full_dispatch_request()
        return response

if __name__ == '__main__':
    app.run(debug=True)