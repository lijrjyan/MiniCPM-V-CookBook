// Plain-HTTP, loopback-only variant of o45-frontend/serve-prod.mjs (same proxy rules).
// http://localhost is a secure context, so getUserMedia works through an ssh -L tunnel.
// usage: node serve_http.mjs --dist <dir> --port 8088 --backend 8021 --livekit 7880
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const arg = (name, dflt) => { const i = process.argv.indexOf(`--${name}`); return i !== -1 ? process.argv[i + 1] : dflt; };
const PORT = parseInt(arg('port', '8088'));
const HOST = arg('host', '127.0.0.1');
const BACKEND = parseInt(arg('backend', '8021'));
const LIVEKIT = parseInt(arg('livekit', '7880'));
const DIST = path.resolve(arg('dist', 'dist'));
const MIME = { '.html': 'text/html', '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.gif': 'image/gif', '.svg': 'image/svg+xml', '.ico': 'image/x-icon', '.woff': 'font/woff', '.woff2': 'font/woff2',
  '.ttf': 'font/ttf', '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.mp4': 'video/mp4', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.txt': 'text/plain' };

function proxy(req, res, port) {
  const up = http.request({ hostname: '127.0.0.1', port, path: req.url, method: req.method, headers: { ...req.headers, host: `127.0.0.1:${port}` } }, (r) => {
    res.writeHead(r.statusCode, r.headers); r.pipe(res);
  });
  up.on('error', (e) => { res.writeHead(502, { 'Content-Type': 'application/json' }); res.end(JSON.stringify({ error: 'Bad Gateway', detail: e.message })); });
  req.pipe(up);
}
function upgrade(req, socket, head, port) {
  const up = http.request({ hostname: '127.0.0.1', port, path: req.url, method: 'GET', headers: { ...req.headers, host: `127.0.0.1:${port}` } });
  up.on('upgrade', (r, s, h) => {
    socket.write('HTTP/1.1 101 Switching Protocols\r\n' + Object.entries(r.headers).map(([k, v]) => `${k}: ${v}`).join('\r\n') + '\r\n\r\n');
    if (h.length) socket.write(h);
    s.pipe(socket); socket.pipe(s);
    s.on('error', () => socket.destroy()); socket.on('error', () => s.destroy());
  });
  up.on('error', () => socket.end());
  if (head && head.length) up.write(head);
  up.end();
}
function serveStatic(req, res) {
  let file = path.join(DIST, req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0]));
  if (!file.startsWith(DIST)) { res.writeHead(403); return res.end(); }
  if (!path.extname(file)) file = path.join(DIST, 'index.html');
  fs.readFile(file, (err, data) => {
    if (err) return fs.readFile(path.join(DIST, 'index.html'), (e2, d2) => { if (e2) { res.writeHead(404); return res.end('Not Found'); } res.writeHead(200, { 'Content-Type': 'text/html' }); res.end(d2); });
    const ext = path.extname(file).toLowerCase();
    res.writeHead(200, { 'Content-Type': MIME[ext] || 'application/octet-stream', 'Cache-Control': ext === '.html' ? 'no-cache' : 'public, max-age=3600' });
    res.end(data);
  });
}
const server = http.createServer((req, res) => {
  if (req.url.startsWith('/api') || req.url.startsWith('/ws') || req.url.startsWith('/download')) return proxy(req, res, BACKEND);
  if (req.url.startsWith('/rtc')) return proxy(req, res, LIVEKIT);
  serveStatic(req, res);
});
server.on('upgrade', (req, socket, head) => {
  if (req.url.startsWith('/rtc')) upgrade(req, socket, head, LIVEKIT);
  else if (req.url.startsWith('/ws')) upgrade(req, socket, head, BACKEND);
  else socket.end();
});
server.listen(PORT, HOST, () => console.log(`frontend http://${HOST}:${PORT}/  /api->${BACKEND}  /rtc->${LIVEKIT}  dist=${DIST}`));
