console.log('=== API Local Test ===\n');

// APIハンドラーをインポート
import apiIndex from './api/index.js';
import apiTest from './api/test.js';
import apiRaces from './api/races.js';
import apiRaceDetail from './api/race/[id].js';

// モックのリクエスト/レスポンスオブジェクト
function createMockReqRes(url, query = {}) {
  const req = {
    method: 'GET',
    url: url,
    query: query
  };
  
  const res = {
    statusCode: null,
    data: null,
    status: function(code) {
      this.statusCode = code;
      return this;
    },
    json: function(data) {
      this.data = data;
      console.log(`Response (${this.statusCode}):`, JSON.stringify(data, null, 2));
      console.log('---\n');
    }
  };
  
  return { req, res };
}

// テスト実行
console.log('1. Testing /api endpoint:');
const { req: req1, res: res1 } = createMockReqRes('/api');
apiIndex(req1, res1);

console.log('2. Testing /api/test endpoint:');
const { req: req2, res: res2 } = createMockReqRes('/api/test');
apiTest(req2, res2);

console.log('3. Testing /api/races endpoint:');
const { req: req3, res: res3 } = createMockReqRes('/api/races', { date: '20250716' });
apiRaces(req3, res3);

console.log('4. Testing /api/race/[id] endpoint:');
const { req: req4, res: res4 } = createMockReqRes('/api/race/R001', { id: 'R001' });
apiRaceDetail(req4, res4);

console.log('=== Test Complete ===');