import { createReadStream, existsSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize } from 'node:path';

const root = join(process.cwd(), 'dist');
const types = {
  '.css': 'text/css',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript',
  '.mjs': 'text/javascript',
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.xml': 'application/xml',
};
createServer((request, response) => {
  const pathname =
    decodeURIComponent(
      new URL(request.url ?? '/', 'http://localhost').pathname,
    ).replace(/^\/sketchsense/, '') || '/';
  let path = normalize(join(root, pathname));
  if (!path.startsWith(root)) {
    response.writeHead(403).end();
    return;
  }
  if (existsSync(path) && statSync(path).isDirectory())
    path = join(path, 'index.html');
  const found = existsSync(path);
  if (!found) path = join(root, '404.html');
  response.writeHead(found ? 200 : 404, {
    'Content-Type': types[extname(path)] ?? 'application/octet-stream',
  });
  createReadStream(path).pipe(response);
}).listen(4321, '127.0.0.1');
