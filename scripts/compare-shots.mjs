import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { PNG } from 'pngjs';
import pixelmatch from 'pixelmatch';

const BASELINE_DIR = process.argv[2] ?? 'parity-baseline';
const AFTER_DIR = process.argv[3] ?? 'parity-after';
const DIFF_DIR = process.argv[4] ?? 'parity-diff';

mkdirSync(DIFF_DIR, { recursive: true });

function cropTopLeft(png, w, h) {
  const out = new PNG({ width: w, height: h });
  for (let y = 0; y < h; y++) {
    const src = y * png.width * 4;
    const dst = y * w * 4;
    out.data.set(png.data.subarray(src, src + w * 4), dst);
  }
  return out;
}

const results = [];
for (const file of readdirSync(BASELINE_DIR).filter((f) => f.endsWith('.png')).sort()) {
  const afterPath = path.join(AFTER_DIR, file);
  try {
    const a = PNG.sync.read(readFileSync(path.join(BASELINE_DIR, file)));
    const b = PNG.sync.read(readFileSync(afterPath));
    const w = Math.min(a.width, b.width);
    const h = Math.min(a.height, b.height);
    const aC = a.width === w && a.height === h ? a : cropTopLeft(a, w, h);
    const bC = b.width === w && b.height === h ? b : cropTopLeft(b, w, h);
    const diff = new PNG({ width: w, height: h });
    const mismatched = pixelmatch(aC.data, bC.data, diff.data, w, h, {
      threshold: 0.1,
      includeAA: false,
    });
    writeFileSync(path.join(DIFF_DIR, file), PNG.sync.write(diff));
    const pct = ((mismatched / (w * h)) * 100).toFixed(2);
    results.push({
      file,
      dims: `${a.width}x${a.height} -> ${b.width}x${b.height}`,
      mismatchPct: Number(pct),
    });
  } catch (err) {
    results.push({ file, error: String(err) });
  }
}

console.table(results);
writeFileSync(path.join(DIFF_DIR, 'report.json'), JSON.stringify(results, null, 2));
const bad = results.filter((r) => r.error || r.mismatchPct > 3);
console.log(`\n${results.length - bad.length}/${results.length} within 3% threshold`);
if (bad.length) {
  console.log('Needs review:');
  for (const r of bad) console.log(` - ${r.file} ${r.error ? r.error : r.mismatchPct + '%'}`);
}
