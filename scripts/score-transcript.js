// Word error rate (WER) and speaker accuracy of a transcript against a corrected reference.
// Usage: node scripts/score-transcript.js <reference.docx|txt> <transcript.docx|txt> [--edits]
// Both files: Kalam .docx transcripts, or text with lines like "[HH:MM:SS] Speaker N: text"
// (other lines ignored) or plain text.
// Disfluency-insensitive: "uh"/"um" and immediate repeats ("in in in") are ignored.
const fs = require("fs");
const zlib = require("zlib");
const [refFile, hypFile, flag] = process.argv.slice(2);

function load(file) {
  const words = [];
  const content = /\.docx$/i.test(file) ? docxText(file) : fs.readFileSync(file, "utf8");
  for (const line of content.replace(/^\uFEFF/, "").split(/\r?\n/)) {
    let m = line.match(/^\[\d\d:\d\d:\d\d\]\s+(?:(Speaker \d+|You|Others):\s*)?(.*)$/);
    let speaker = null, text;
    if (m) { speaker = m[1] || null; text = m[2]; }
    else if (/^\s*$/.test(line) || /^\W*(Kalam Transcript\s*$|(Date|Duration|Sources|Recording|Engine|Transcript):)/.test(line)) continue;
    else text = line;
    for (const w of normalise(text)) words.push({ w, speaker });
  }
  // Disfluency-insensitive: drop fillers and immediate repeats ("in in in" -> "in").
  return words.filter((x, k) => !/^(uh|um|uhm|erm|hmm|mm)$/.test(x.w) && !(k > 0 && words[k - 1].w === x.w));
}

// Paragraph text of a .docx (one line per paragraph), read straight from the zip.
function docxText(file) {
  const zip = fs.readFileSync(file);
  let eocd = zip.length - 22;
  while (eocd >= 0 && zip.readUInt32LE(eocd) !== 0x06054b50) eocd--;
  let p = zip.readUInt32LE(eocd + 16);
  for (let n = zip.readUInt16LE(eocd + 10); n > 0; n--) {
    const method = zip.readUInt16LE(p + 10), size = zip.readUInt32LE(p + 20);
    const nameLen = zip.readUInt16LE(p + 28), extra = zip.readUInt16LE(p + 30), comment = zip.readUInt16LE(p + 32);
    if (zip.toString("utf8", p + 46, p + 46 + nameLen) === "word/document.xml") {
      const local = zip.readUInt32LE(p + 42);
      const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
      const raw = zip.subarray(start, start + size);
      const xml = (method === 8 ? zlib.inflateRawSync(raw) : raw).toString("utf8");
      const decode = (t) => t.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&apos;/g, "'").replace(/&amp;/g, "&");
      return xml.split("</w:p>").map((para) => decode([...para.matchAll(/<w:t(?: [^>]*)?>([^<]*)<\/w:t>/g)].map((m) => m[1]).join(""))).join("\n");
    }
    p += 46 + nameLen + extra + comment;
  }
  throw new Error("not a .docx: " + file);
}

function normalise(text) {
  return text.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[-\u2013\u2014/]/g, " ")
    .replace(/[^a-z0-9'\u00c0-\u024f ]/g, " ")
    .split(/\s+/)
    .map((w) => w.replace(/^'+|'+$/g, ""))
    .filter(Boolean);
}

const ref = load(refFile), hyp = load(hypFile);
const n = ref.length, m = hyp.length;
// Levenshtein with backtrace.
const d = Array.from({ length: n + 1 }, () => new Uint32Array(m + 1));
for (let i = 0; i <= n; i++) d[i][0] = i;
for (let j = 0; j <= m; j++) d[0][j] = j;
for (let i = 1; i <= n; i++)
  for (let j = 1; j <= m; j++)
    d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1].w === hyp[j - 1].w ? 0 : 1));
let i = n, j = m, S = 0, D = 0, I = 0;
const ops = [];
while (i > 0 || j > 0) {
  if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (ref[i - 1].w === hyp[j - 1].w ? 0 : 1)) {
    const same = ref[i - 1].w === hyp[j - 1].w;
    if (!same) S++;
    ops.push({ t: same ? "=" : "S", r: ref[i - 1], h: hyp[j - 1] });
    i--; j--;
  } else if (i > 0 && d[i][j] === d[i - 1][j] + 1) { D++; ops.push({ t: "D", r: ref[i - 1] }); i--; }
  else { I++; ops.push({ t: "I", h: hyp[j - 1] }); j--; }
}
ops.reverse();

const wer = (S + D + I) / n;
console.log(`words: ref ${n}, hyp ${m}`);
console.log(`WER ${(100 * wer).toFixed(1)}%  (sub ${S}, del ${D}, ins ${I})`);

// Speaker accuracy over aligned words, with the best one-to-one label mapping.
const pairs = ops.filter((o) => o.r && o.h && o.r.speaker && o.h.speaker);
if (pairs.length) {
  const counts = {};
  for (const o of pairs) counts[o.h.speaker + "|" + o.r.speaker] = (counts[o.h.speaker + "|" + o.r.speaker] || 0) + 1;
  const hs = [...new Set(pairs.map((o) => o.h.speaker))], rs = [...new Set(pairs.map((o) => o.r.speaker))];
  let best = 0, bestMap = null;
  const perm = (left, used, acc, map) => {
    if (!left.length) { if (acc > best) { best = acc; bestMap = { ...map }; } return; }
    const [h, ...rest] = left;
    perm(rest, used, acc, map); // unmapped
    for (const r of rs) if (!used.has(r)) {
      used.add(r); map[h] = r;
      perm(rest, used, acc + (counts[h + "|" + r] || 0), map);
      used.delete(r); delete map[h];
    }
  };
  perm(hs, new Set(), 0, {});
  console.log(`speaker: ${(100 * best / pairs.length).toFixed(1)}% of aligned words correctly attributed  map ${JSON.stringify(bestMap)}`);
  const wrong = [];
  let run = null;
  for (const o of pairs) {
    const ok = bestMap[o.h.speaker] === o.r.speaker;
    if (!ok) { if (!run) { run = { h: o.h.speaker, r: o.r.speaker, words: [] }; wrong.push(run); } run.words.push(o.r.w); }
    else run = null;
  }
  for (const w of wrong) console.log(`  wrong speaker (${w.h} for ${w.r}): ${w.words.slice(0, 14).join(" ")}${w.words.length > 14 ? " …" : ""}`);
}

if (flag === "--edits") {
  // Group consecutive errors with one word of context.
  const out = [];
  for (let k = 0; k < ops.length; k++) {
    if (ops[k].t === "=") continue;
    let e = k;
    while (e + 1 < ops.length && ops[e + 1].t !== "=") e++;
    const r = ops.slice(k, e + 1).filter((o) => o.r).map((o) => o.r.w).join(" ");
    const h = ops.slice(k, e + 1).filter((o) => o.h).map((o) => o.h.w).join(" ");
    out.push(`  "${r}" -> "${h}"`);
    k = e;
  }
  console.log(out.join("\n"));
}
