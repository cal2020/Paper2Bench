// A loopback-only static fixture server; it has no API or model runtime.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '../deployment/demo/dist');
const port = Number(process.env.P2B_DEMO_TEST_PORT || '4173');
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error('Invalid demo test port');
}
if (!fs.existsSync(path.join(root, 'index.html')) || !fs.existsSync(path.join(root, 'asset-manifest.json'))) {
  throw new Error('Build the demo before running browser checks; deployment/demo/dist is missing');
}

const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.webp': 'image/webp',
  '.ico': 'image/x-icon',
  '.zip': 'application/zip',
};

const server = http.createServer((request, response) => {
  const send = (status, text) => {
    response.writeHead(status, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
    response.end(text);
  };
  if (request.method !== 'GET' && request.method !== 'HEAD') return send(405, 'Static fixture only');
  let name;
  try { name = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname); }
  catch { return send(400, 'Invalid asset path'); }
  if (name === '/') name = '/index.html';
  const relative = name.replace(/^\/+/, '');
  if (!relative || relative.split('/').some(part => part.startsWith('.') || part.includes('\\'))) {
    return send(403, 'Invalid asset path');
  }
  const file = path.resolve(root, relative);
  if (!file.startsWith(root + path.sep)) return send(403, 'Outside fixture');
  let stat;
  try {
    const actual = fs.realpathSync(file);
    if (actual !== file) return send(403, 'Symlink fixture path');
    stat = fs.statSync(file);
  } catch { return send(404, 'Unknown fixture asset'); }
  if (!stat.isFile() || stat.size > 16 * 1024 * 1024) return send(404, 'Unknown fixture asset');
  response.writeHead(200, {
    'Content-Type': types[path.extname(file)] || 'application/octet-stream',
    'Content-Length': stat.size,
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
  });
  if (request.method === 'HEAD') return response.end();
  const stream = fs.createReadStream(file);
  stream.on('error', () => response.destroy());
  stream.pipe(response);
});

server.listen(port, '127.0.0.1', () => console.log(`Demo fixture: http://127.0.0.1:${port}`));
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, () => server.close(() => process.exit(0)));
}
