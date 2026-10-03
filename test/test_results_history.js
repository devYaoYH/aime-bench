/**
 * Offline checks for the results plot's chronological Pareto frontier.
 * Run `node test/test_results_history.js` from the repository root. The VM loads
 * the real viewer code without a browser or network; tests cover non-improving
 * attempts, ties, invalid evidence, initialization timestamps, and SVG axes.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const context = vm.createContext({document:{addEventListener(){}}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../src/viewer/results/viewer.js'),'utf8'),context);
function row(id, start, latency, official=start) {
  return {id,attempt_started_at_utc:start,started_at_utc:official,time_to_18_s:latency,
    metadata:{label:id,intervention:{label:'test intervention'}}};
}
function history(rows) {
  context.rows=rows;
  return JSON.parse(JSON.stringify(vm.runInContext('attemptHistory(rows)',context)));
}
const rows=[row('slow','2026-10-03T20:40:00Z',300),
  row('first','2026-10-03T20:00:00Z',100,'2026-10-03T20:59:00Z'),
  row('best','2026-10-03T20:30:00Z',70),row('tie','2026-10-03T20:31:00Z',70),
  row('unmet','2026-10-03T20:50:00Z',null),row('missing','bad timestamp',60),
  row('invalid','2026-10-03T20:10:00Z',NaN),row('negative','2026-10-03T20:15:00Z',-1)];
const before=rows.map(r=>r.id);
const result=history(rows);
assert.deepEqual(result.points.map(r=>r.id),['first','best','tie','slow']);
assert.deepEqual(result.frontier.map(r=>r.id),['first','best']);
assert.deepEqual(rows.map(r=>r.id),before,'Sorting must not mutate the inventory');
assert.deepEqual(history([]),{points:[],frontier:[],baselines:[]});
assert.equal(history([row('legacy',null,54,'2026-10-03T20:00:00Z')]).points.length,1);
context.rows=rows;
vm.runInContext('results={reference_floor_s:54}',context);
const svg=vm.runInContext('comparisonPlot(attemptHistory(rows))',context);
assert.equal((svg.match(/class="attempt-dot"/g)||[]).length,4);
assert.match(svg,/class="pareto-frontier" d="M [^"]+ H [^"]+ V [^"]+ H [^"]+"/);
const floor=svg.match(/class="floor-line" x1="([^"]+)" x2="([^"]+)" y1="([^"]+)" y2="([^"]+)"/);
assert.ok(floor);assert.notEqual(floor[1],floor[2]);assert.equal(floor[3],floor[4]);
assert.match(svg,/Attempt started · America\/Los_Angeles/);
assert.match(svg,/Time to 18 verified correct answers \(seconds\)/);
context.rows=[row('single','2026-10-03T20:00:00Z',54)];
assert.doesNotMatch(vm.runInContext('comparisonPlot(attemptHistory(rows))',context),/NaN|Infinity/);
context.rows=[row('first','2026-10-03T20:07:00Z',92),
  row('20261003T202152.418590Z','2026-10-03T20:21:00Z',336),
  row('20261003T203338.063138Z','2026-10-03T20:33:00Z',517),
  row('improved','2026-10-03T20:53:00Z',85),row('retry','2026-10-03T21:02:00Z',97)];
const filtered=history(context.rows);
assert.deepEqual(filtered.points.map(r=>r.plot_number),[1,4,5]);
assert.deepEqual(filtered.baselines.map(r=>r.plot_number),[2,3]);
assert.deepEqual(filtered.frontier.map(r=>r.id),['first','improved']);
const filteredSvg=vm.runInContext('comparisonPlot(attemptHistory(rows))',context);
assert.equal((filteredSvg.match(/class="attempt-dot"/g)||[]).length,3);
assert.doesNotMatch(filteredSvg,/20261003T202152|20261003T203338/);
assert.match(filteredSvg,/>100s<\/text>/,'Y-axis must rescale around the remaining runs');
assert.match(filteredSvg,/>50s<\/text>/,'Y-axis must start at 50 seconds');
assert.doesNotMatch(filteredSvg,/>0s<\/text>/);
const filteredFloor=filteredSvg.match(/class="floor-line"[^>]* y1="([^"]+)"/);
assert.ok(Math.abs(Number(filteredFloor[1])-(330-4/50*290))<1e-9,'54-second floor must use the shifted scale');
console.log('Results history checks passed: chronological points, strict running minimum, timestamps, floor, and SVG.');
