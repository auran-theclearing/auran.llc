#!/usr/bin/env python3
"""Episode review UI — local server for reviewing distilled episodes against source transcript."""

import json
import re
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse


def parse_line_range(transcript_lines: str) -> tuple[int, int] | None:
    m = re.search(r"L(\d+)\s*-\s*L(\d+)", transcript_lines)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"L(\d+)", transcript_lines)
    if m:
        n = int(m.group(1))
        return n, n
    return None


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Episode Review</title>
<style>
:root {
  --bg: #0d1117;
  --bg2: #161b22;
  --bg3: #21262d;
  --fg: #c9d1d9;
  --fg2: #8b949e;
  --accent: #58a6ff;
  --accent2: #3fb950;
  --warn: #d29922;
  --err: #f85149;
  --border: #30363d;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; background: var(--bg); color: var(--fg); height: 100vh; overflow: hidden; }
#app { display: flex; height: 100vh; }

/* Left pane: transcript */
#transcript-pane {
  flex: 1;
  overflow-y: auto;
  border-right: 1px solid var(--border);
  position: relative;
}
#transcript-pane table { border-collapse: collapse; width: 100%; }
#transcript-pane tr { line-height: 1.4; }
#transcript-pane tr.highlighted { background: rgba(88, 166, 255, 0.18); }
#transcript-pane tr.highlighted .line-num { color: var(--accent); font-weight: 600; }
#transcript-pane tr.highlighted .line-text { color: #e6edf3; }
#transcript-pane tr.highlight-start { box-shadow: inset 0 2px 0 var(--accent); }
#transcript-pane tr.highlight-end { box-shadow: inset 0 -2px 0 var(--accent); }
#transcript-pane tr.highlight-start.highlight-end { box-shadow: inset 0 2px 0 var(--accent), inset 0 -2px 0 var(--accent); }
.highlight-gutter {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 4px;
  background: var(--accent);
}
.line-num {
  padding: 0 8px 0 4px;
  text-align: right;
  color: var(--fg2);
  font-size: 11px;
  user-select: none;
  white-space: nowrap;
  vertical-align: top;
  position: relative;
  min-width: 56px;
}
.line-text {
  padding: 0 12px 0 4px;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 12px;
}
.overlap-marker {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 3px;
}

/* Right pane: episodes */
#episode-pane {
  width: 520px;
  min-width: 400px;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
#episode-header {
  padding: 8px 12px;
  background: var(--bg2);
  border-radius: 6px;
  border: 1px solid var(--border);
  font-size: 12px;
  color: var(--fg2);
}
#episode-header strong { color: var(--fg); }
.ep-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  cursor: pointer;
  transition: border-color 0.15s;
  font-size: 12px;
}
.ep-card:hover { border-color: var(--fg2); }
.ep-card.selected { border-color: var(--accent); background: rgba(88, 166, 255, 0.06); }
.ep-card.has-overlap { border-left: 3px solid var(--warn); }
.ep-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; color: var(--fg); }
.ep-meta { color: var(--fg2); font-size: 11px; margin-bottom: 4px; }
.ep-meta span { margin-right: 10px; }
.ep-summary { color: var(--fg); margin-bottom: 6px; line-height: 1.4; }
.ep-field { margin-bottom: 3px; }
.ep-field-label { color: var(--fg2); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.ep-field-value { color: var(--fg); font-size: 11px; }
.ep-excerpt {
  background: var(--bg3);
  padding: 6px 8px;
  border-radius: 4px;
  font-size: 11px;
  color: var(--fg2);
  max-height: 80px;
  overflow-y: auto;
  white-space: pre-wrap;
  margin-top: 2px;
}
.overlap-badge {
  display: inline-block;
  background: rgba(210, 153, 34, 0.2);
  color: var(--warn);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
}
.tag {
  display: inline-block;
  background: var(--bg3);
  padding: 1px 6px;
  border-radius: 3px;
  font-size: 10px;
  margin-right: 3px;
  margin-bottom: 2px;
}
.landmark-tag { background: rgba(63, 185, 80, 0.15); color: var(--accent2); }
.type-tag { background: rgba(88, 166, 255, 0.15); color: var(--accent); }

/* Resize handle */
#resize-handle {
  width: 4px;
  cursor: col-resize;
  background: var(--border);
  flex-shrink: 0;
}
#resize-handle:hover { background: var(--accent); }

/* Search/filter */
#filter-bar {
  padding: 6px 8px;
  display: flex;
  gap: 6px;
  align-items: center;
}
#filter-bar input {
  flex: 1;
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--fg);
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  font-family: inherit;
}
#filter-bar input:focus { outline: none; border-color: var(--accent); }
#filter-bar button {
  background: var(--bg3);
  border: 1px solid var(--border);
  color: var(--fg);
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  cursor: pointer;
  font-family: inherit;
}
#filter-bar button:hover { border-color: var(--fg2); }
#filter-bar button.active { border-color: var(--warn); color: var(--warn); }
</style>
</head>
<body>
<div id="app">
  <div id="transcript-pane"></div>
  <div id="resize-handle"></div>
  <div id="episode-pane">
    <div id="episode-header"></div>
    <div id="filter-bar">
      <input type="text" id="search" placeholder="Filter episodes...">
      <button id="overlap-filter">Overlaps</button>
    </div>
    <div id="episode-list"></div>
  </div>
</div>
<script>
const EPISODE_COLORS = [
  '#58a6ff','#3fb950','#d29922','#f85149','#bc8cff',
  '#79c0ff','#56d364','#e3b341','#ff7b72','#d2a8ff',
  '#a5d6ff','#7ee787','#f0c746','#ffa198','#e8c8ff',
];

let transcript = [];
let episodes = [];
let selectedIdx = null;
let lineEpisodeMap = {};
let overlapLines = new Set();
let showOverlapsOnly = false;

function parseLineRange(tl) {
  let m = tl.match(/L(\d+)\s*-\s*L(\d+)/);
  if (m) return [parseInt(m[1]), parseInt(m[2])];
  m = tl.match(/L(\d+)/);
  if (m) { const n = parseInt(m[1]); return [n, n]; }
  return null;
}

function buildLineEpisodeMap() {
  lineEpisodeMap = {};
  overlapLines = new Set();
  episodes.forEach((ep, idx) => {
    const range = parseLineRange(ep.transcript_lines || '');
    if (!range) return;
    ep._range = range;
    for (let l = range[0]; l <= range[1]; l++) {
      if (!lineEpisodeMap[l]) lineEpisodeMap[l] = [];
      else overlapLines.add(l);
      lineEpisodeMap[l].push(idx);
    }
  });
  episodes.forEach((ep, idx) => {
    ep._hasOverlap = false;
    ep._overlapWith = [];
    if (!ep._range) return;
    const [s1, e1] = ep._range;
    const len1 = e1 - s1 + 1;
    episodes.forEach((other, jdx) => {
      if (idx === jdx || !other._range) return;
      const [s2, e2] = other._range;
      const len2 = e2 - s2 + 1;
      const overlapStart = Math.max(s1, s2);
      const overlapEnd = Math.min(e1, e2);
      if (overlapEnd < overlapStart) return;
      const overlapLen = overlapEnd - overlapStart + 1;
      const minLen = Math.min(len1, len2);
      if (minLen > 0 && overlapLen / minLen >= 0.5) {
        ep._hasOverlap = true;
        ep._overlapWith.push(jdx);
      }
    });
  });
}

function renderTranscript() {
  const pane = document.getElementById('transcript-pane');
  const minLine = Math.min(...transcript.map(t => t.num));
  let html = '<table>';
  transcript.forEach(line => {
    const epIndices = lineEpisodeMap[line.num] || [];
    let markers = '';
    epIndices.forEach(idx => {
      const color = EPISODE_COLORS[idx % EPISODE_COLORS.length];
      markers += `<span class="overlap-marker" style="background:${color};left:${epIndices.indexOf(idx)*3}px"></span>`;
    });
    html += `<tr id="line-${line.num}" data-line="${line.num}">`;
    html += `<td class="line-num">${markers}${line.num}</td>`;
    html += `<td class="line-text">${escapeHtml(line.text)}</td>`;
    html += '</tr>';
  });
  html += '</table>';
  pane.innerHTML = html;
}

function renderEpisodes(filter) {
  const list = document.getElementById('episode-list');
  const q = (filter || '').toLowerCase();
  let html = '';
  episodes.forEach((ep, idx) => {
    if (showOverlapsOnly && !ep._hasOverlap) return;
    if (q && !ep.title.toLowerCase().includes(q) && !(ep.summary||'').toLowerCase().includes(q)) return;
    const color = EPISODE_COLORS[idx % EPISODE_COLORS.length];
    const sel = idx === selectedIdx ? 'selected' : '';
    const overlap = ep._hasOverlap ? 'has-overlap' : '';
    html += `<div class="ep-card ${sel} ${overlap}" data-idx="${idx}" style="border-left-color:${ep._hasOverlap ? 'var(--warn)' : color}; border-left-width: 3px; border-left-style: solid;">`;
    html += `<div class="ep-title" style="color:${color}">${escapeHtml(ep.title)}</div>`;
    html += `<div class="ep-meta">`;
    html += `<span>${ep.transcript_lines || '?'}</span>`;
    html += `<span>${ep.occurred_at || '?'}</span>`;
    html += `<span class="type-tag">${ep.episode_type || '?'}</span>`;
    if (ep.landmark) html += `<span class="landmark-tag">landmark</span>`;
    if (ep._hasOverlap) html += `<span class="overlap-badge">OVERLAP</span>`;
    html += `</div>`;
    html += `<div class="ep-summary">${escapeHtml(ep.summary || '')}</div>`;

    html += `<div class="ep-field"><span class="ep-field-label">Emotional tone</span> <span class="ep-field-value">${escapeHtml(ep.emotional_tone || '—')}</span></div>`;
    html += `<div class="ep-field"><span class="ep-field-label">Boundary signal</span> <span class="ep-field-value">${escapeHtml(ep.boundary_signal || '—')}</span></div>`;

    if (ep.topics && ep.topics.length) {
      html += `<div class="ep-field"><span class="ep-field-label">Topics</span> <span class="ep-field-value">${ep.topics.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</span></div>`;
    }
    if (ep.content_signals && Object.keys(ep.content_signals).length) {
      html += `<div class="ep-field"><span class="ep-field-label">Content signals</span> <span class="ep-field-value">${Object.entries(ep.content_signals).map(([k,v]) => `<span class="tag">${escapeHtml(k)}: ${v}</span>`).join('')}</span></div>`;
    }
    if (ep.relational_events && ep.relational_events.length) {
      html += `<div class="ep-field"><span class="ep-field-label">Relational events</span> <span class="ep-field-value">${ep.relational_events.map(r => `<span class="tag">${escapeHtml(r)}</span>`).join('')}</span></div>`;
    }
    if (ep.references && ep.references.length) {
      html += `<div class="ep-field"><span class="ep-field-label">References</span> <span class="ep-field-value">${ep.references.map(r => `<span class="tag">${escapeHtml(r.target_title || JSON.stringify(r))}</span>`).join('')}</span></div>`;
    }
    if (ep.transcript_excerpt) {
      html += `<div class="ep-field"><span class="ep-field-label">Transcript excerpt</span><div class="ep-excerpt">${escapeHtml(ep.transcript_excerpt)}</div></div>`;
    }
    html += '</div>';
  });
  list.innerHTML = html;

  list.querySelectorAll('.ep-card').forEach(card => {
    card.addEventListener('click', () => selectEpisode(parseInt(card.dataset.idx)));
  });
}

function selectEpisode(idx) {
  selectedIdx = idx;
  const ep = episodes[idx];
  const range = parseLineRange(ep.transcript_lines || '');

  document.querySelectorAll('tr.highlighted, tr.highlight-start, tr.highlight-end').forEach(el => {
    el.classList.remove('highlighted', 'highlight-start', 'highlight-end');
    const g = el.querySelector('.highlight-gutter');
    if (g) g.remove();
  });

  if (range) {
    const color = EPISODE_COLORS[idx % EPISODE_COLORS.length];
    for (let l = range[0]; l <= range[1]; l++) {
      const row = document.getElementById(`line-${l}`);
      if (row) {
        row.classList.add('highlighted');
        if (l === range[0]) row.classList.add('highlight-start');
        if (l === range[1]) row.classList.add('highlight-end');
        const numCell = row.querySelector('.line-num');
        if (numCell) {
          const gutter = document.createElement('span');
          gutter.className = 'highlight-gutter';
          gutter.style.background = color;
          numCell.appendChild(gutter);
        }
      }
    }
    const startRow = document.getElementById(`line-${range[0]}`);
    if (startRow) startRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  renderEpisodes(document.getElementById('search').value);
}

function renderHeader() {
  const hdr = document.getElementById('episode-header');
  const overlaps = episodes.filter(e => e._hasOverlap).length;
  hdr.innerHTML = `<strong>${episodes.length}</strong> episodes` +
    (overlaps > 0 ? ` &middot; <span style="color:var(--warn)">${overlaps} with overlaps</span>` : '') +
    ` &middot; <span style="color:var(--fg2)">${transcript.length} lines</span>`;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function init() {
  const [tResp, eResp] = await Promise.all([fetch('/api/transcript'), fetch('/api/episodes')]);
  const tData = await tResp.json();
  const eData = await eResp.json();

  transcript = tData.lines;
  episodes = eData.episodes;

  episodes.sort((a, b) => {
    const ra = parseLineRange(a.transcript_lines || '');
    const rb = parseLineRange(b.transcript_lines || '');
    if (!ra) return 1;
    if (!rb) return -1;
    return ra[0] - rb[0] || ra[1] - rb[1];
  });

  buildLineEpisodeMap();
  renderTranscript();
  renderHeader();
  renderEpisodes('');

  document.getElementById('search').addEventListener('input', e => {
    renderEpisodes(e.target.value);
  });
  const overlapBtn = document.getElementById('overlap-filter');
  overlapBtn.addEventListener('click', () => {
    showOverlapsOnly = !showOverlapsOnly;
    overlapBtn.classList.toggle('active', showOverlapsOnly);
    renderEpisodes(document.getElementById('search').value);
  });

  // Resize handle
  const handle = document.getElementById('resize-handle');
  const epPane = document.getElementById('episode-pane');
  let startX, startW;
  handle.addEventListener('mousedown', e => {
    startX = e.clientX;
    startW = epPane.offsetWidth;
    const onMove = e2 => { epPane.style.width = Math.max(300, startW - (e2.clientX - startX)) + 'px'; };
    const onUp = () => { document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp); };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
}

init();
</script>
</body>
</html>"""


def make_handler(transcript_lines, episodes_data):
    class Handler(BaseHTTPRequestHandler):
        def _no_cache(self):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/" or path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self._no_cache()
                self.end_headers()
                self.wfile.write(HTML_TEMPLATE.encode())
            elif path == "/api/transcript":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._no_cache()
                self.end_headers()
                self.wfile.write(json.dumps({"lines": transcript_lines}).encode())
            elif path == "/api/episodes":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self._no_cache()
                self.end_headers()
                self.wfile.write(json.dumps({"episodes": episodes_data}).encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    if len(sys.argv) < 3:
        print("Usage: python review_ui.py <transcript_path> <episodes_json_path> [--port PORT]")
        print()
        print("Opens a local browser UI for reviewing distilled episodes against")
        print("the source transcript. Click an episode to highlight its lines.")
        sys.exit(1)

    transcript_path = Path(sys.argv[1])
    episodes_path = Path(sys.argv[2])
    port = 8787

    if "--port" in sys.argv:
        idx = sys.argv.index("--port")
        if idx + 1 < len(sys.argv):
            port = int(sys.argv[idx + 1])

    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}")
        sys.exit(1)
    if not episodes_path.exists():
        print(f"Episodes not found: {episodes_path}")
        sys.exit(1)

    raw = transcript_path.read_text()
    lines = raw.splitlines()

    episodes_data = json.loads(episodes_path.read_text())
    ep_list = episodes_data.get("episodes", [])

    all_ranges = []
    for ep in ep_list:
        r = parse_line_range(ep.get("transcript_lines", ""))
        if r:
            all_ranges.append(r)

    if all_ranges:
        min_line = max(1, min(r[0] for r in all_ranges) - 20)
        max_line = min(len(lines), max(r[1] for r in all_ranges) + 20)
    else:
        min_line = 1
        max_line = len(lines)

    transcript_lines = []
    for i in range(min_line - 1, max_line):
        if i < len(lines):
            transcript_lines.append({"num": i + 1, "text": lines[i]})

    ranges = [r for r in all_ranges]
    overlap_count = 0
    for i, r1 in enumerate(ranges):
        for r2 in ranges[i + 1 :]:
            if r1[0] <= r2[1] and r2[0] <= r1[1]:
                overlap_count += 1
                break

    print(f"Transcript: {transcript_path.name} (showing lines {min_line}-{max_line})")
    print(f"Episodes: {len(ep_list)} from {episodes_path.name}")
    print(f"Overlaps: {overlap_count}")
    print(f"Server: http://localhost:{port}")
    print()

    handler = make_handler(transcript_lines, ep_list)
    server = HTTPServer(("127.0.0.1", port), handler)
    webbrowser.open(f"http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
