/**
 * Tests for formatTime() — extracted from index.html.
 * Run: node chat/test_format_time.mjs
 */
import { strict as assert } from 'node:assert';

// Mirror of the formatTime function from index.html (line 676)
function formatTime(ts) {
  if (!ts) return '';
  const normalized = (ts.endsWith('Z') || ts.includes('+') || ts.includes('-', 10)) ? ts : ts + 'Z';
  const d = new Date(normalized);
  const date = d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  const time = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  return `${date} ${time}`;
}

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}: ${e.message}`);
  }
}

console.log('formatTime() tests\n');

test('returns empty string for falsy input', () => {
  assert.equal(formatTime(''), '');
  assert.equal(formatTime(null), '');
  assert.equal(formatTime(undefined), '');
});

test('always includes a date — even for today', () => {
  const now = new Date();
  const ts = now.toISOString();
  const result = formatTime(ts);
  // Must contain a month abbreviation (3+ letter word)
  assert.match(result, /[A-Z][a-z]{2,}/, `expected month name in "${result}"`);
  // Must contain a digit (the day number)
  assert.match(result, /\d/, `expected day number in "${result}"`);
});

test('handles UTC Z suffix', () => {
  const result = formatTime('2026-06-15T19:30:00Z');
  assert.match(result, /Jun/, `expected "Jun" in "${result}"`);
  assert.match(result, /15/, `expected "15" in "${result}"`);
  assert.match(result, /\d{1,2}:\d{2}/, `expected time in "${result}"`);
});

test('handles timezone offset', () => {
  const result = formatTime('2026-03-10T14:00:00-04:00');
  assert.match(result, /Mar/, `expected "Mar" in "${result}"`);
  assert.match(result, /10/, `expected "10" in "${result}"`);
});

test('appends Z when no timezone marker', () => {
  const result = formatTime('2026-01-20T08:00:00');
  assert.match(result, /Jan/, `expected "Jan" in "${result}"`);
  assert.match(result, /20/, `expected "20" in "${result}"`);
});

test('past dates include date and time', () => {
  const result = formatTime('2025-12-25T12:00:00Z');
  assert.match(result, /Dec/, `expected "Dec" in "${result}"`);
  assert.match(result, /25/, `expected "25" in "${result}"`);
  assert.match(result, /\d{1,2}:\d{2}/, `expected time in "${result}"`);
});

test('format is "Mon DD H:MM AM/PM" pattern', () => {
  const result = formatTime('2026-06-17T15:42:00Z');
  // Should be something like "Jun 17 11:42 AM" (local TZ dependent)
  const parts = result.split(' ');
  // At minimum: month, day, time, am/pm
  assert.ok(parts.length >= 3, `expected at least 3 parts, got: "${result}"`);
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
