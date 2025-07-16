export default function handler(req, res) {
  console.log('=== /api/test Request ===');
  console.log('Method:', req.method);
  console.log('URL:', req.url);
  console.log('========================');

  if (req.method !== 'GET') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

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
}