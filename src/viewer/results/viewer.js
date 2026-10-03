/**
 * Compare recorded time to 18 correct across canonical attempts. Use /results
 * through src.viewer_server after syncing evidence and annotating metadata.json.
 * Plot attempts by start timestamp with a dotted best-so-far frontier, show the
 * serial grader floor, and expose changed
 * controls without treating single-run differences as isolated causal effects.
 */
const $ = selector => document.querySelector(selector);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sec = v => v == null ? '—' : `${v.toFixed(2)}s`;
const colors = ['#147d65','#54729b','#b37b30','#865b89','#b65e46','#67804b'];
const attemptLink = row => `/?attempt=${encodeURIComponent(row.id)}`;
let results;
function attemptHistory(rows) {
  const points=rows.filter(r=>r.time_to_18_s!=null && Number.isFinite(r.time_to_18_s) && r.time_to_18_s>=0)
    .map(r=>({...r,start_ms:Date.parse(r.attempt_started_at_utc??r.started_at_utc)}))
    .filter(r=>Number.isFinite(r.start_ms)).sort((a,b)=>a.start_ms-b.start_ms || a.id.localeCompare(b.id));
  let best=Infinity;
  const frontier=points.filter(r=>{if(r.time_to_18_s<best){best=r.time_to_18_s;return true;}return false;});
  return {points,frontier};
}
const historyDate=new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',month:'short',day:'numeric',hour:'numeric',minute:'2-digit',second:'2-digit',timeZoneName:'short'});
const historyTick=new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',hour:'numeric',minute:'2-digit'});
function comparisonPlot(history) {
  const {points,frontier}=history;
  if (!points.length) return '<div class="chart-empty">No attempt has both a measured time to 18 and a recorded start timestamp.</div>';
  const width=1080,height=410,left=86,right=30,top=40,bottom=80;
  const span=Math.max(points.at(-1).start_ms-points[0].start_ms,60000),padding=span*.035;
  const xmin=points[0].start_ms-padding,xmax=points.at(-1).start_ms+padding;
  const max=Math.ceil(Math.max(60,...points.map(r=>r.time_to_18_s))/60)*60;
  const x=t=>left+(t-xmin)/(xmax-xmin)*(width-left-right);
  const y=t=>height-bottom-t/max*(height-top-bottom);
  let svg=`<svg class="comparison-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Attempt start timestamp versus end-to-end time to 18 verified correct answers, with a dotted best-so-far Pareto step and a horizontal 54 second grader floor"><text x="${left}" y="18">Time to 18 verified correct answers (seconds)</text>`;
  for(let i=0;i<=6;i++){
    const t=max*i/6,stamp=xmin+(xmax-xmin)*i/6;
    svg+=`<line class="grid" x1="${left}" x2="${width-right}" y1="${y(t)}" y2="${y(t)}"/><text text-anchor="end" x="${left-12}" y="${y(t)+4}">${Math.round(t)}s</text><line class="grid" x1="${x(stamp)}" x2="${x(stamp)}" y1="${top}" y2="${height-bottom}"/><text text-anchor="middle" x="${x(stamp)}" y="${height-bottom+26}">${esc(historyTick.format(stamp))}</text>`;
  }
  svg+=`<text text-anchor="middle" x="${left+(width-left-right)/2}" y="${height-12}">Attempt started · America/Los_Angeles · ${esc(new Intl.DateTimeFormat('en-US',{timeZone:'America/Los_Angeles',month:'short',day:'numeric',year:'numeric'}).format(points[0].start_ms))}</text>`;
  const floor=results.reference_floor_s;
  svg+=`<line class="floor-line" x1="${left}" x2="${width-right}" y1="${y(floor)}" y2="${y(floor)}"/><text class="floor-label" x="${left+8}" y="${y(floor)+17}">54s grader floor · 18 × 3s</text>`;
  let d=`M ${x(frontier[0].start_ms)} ${y(frontier[0].time_to_18_s)}`;
  frontier.slice(1).forEach(r=>{d+=` H ${x(r.start_ms)} V ${y(r.time_to_18_s)}`;});
  d+=` H ${x(xmax)}`;
  svg+=`<path class="pareto-frontier" d="${d}"/>`;
  points.forEach((r,i)=>{
    const description=`${r.metadata.label} · ${sec(r.time_to_18_s)} · started ${historyDate.format(r.start_ms)} · ${r.metadata.intervention.label}`;
    svg+=`<a class="attempt-dot-link" href="${attemptLink(r)}" aria-label="${esc(description)}"><title>${esc(description)}</title><circle class="attempt-dot" data-attempt="${esc(r.id)}" cx="${x(r.start_ms)}" cy="${y(r.time_to_18_s)}" r="11" fill="${colors[i%colors.length]}"/><text class="dot-number" text-anchor="middle" x="${x(r.start_ms)}" y="${y(r.time_to_18_s)+4}">${i+1}</text></a>`;
  });
  return svg+'</svg>';
}
function historyLegend(history) {
  const bestIds=new Set(history.frontier.map(r=>r.id));
  return history.points.map((r,i)=>`<a class="history-item" href="${attemptLink(r)}"><span class="history-number" style="background:${colors[i%colors.length]}">${i+1}</span><span><strong>${esc(r.metadata.label)}</strong><small>${sec(r.time_to_18_s)} · ${esc(historyDate.format(r.start_ms))}${bestIds.has(r.id)?' · new best':''}</small><small>${esc(r.metadata.intervention.label)}</small></span></a>`).join('');
}
function interventionCards(rows) {
  const comparisons=rows.filter(r=>r.metadata && r.comparison?.saved_s!=null);
  if(!comparisons.length)return '<p class="detail-note">Annotate a reference attempt in metadata.json to compare interventions.</p>';
  return comparisons.map(r=>{const m=r.metadata,c=r.comparison,faster=c.saved_s>=0;return `<article class="intervention-card"><h3>${esc(m.intervention.label)}</h3><small>${esc(m.label)} vs ${esc(c.reference_label)}</small><div class="delta ${faster?'':'slower'}">${sec(Math.abs(c.saved_s))} ${faster?'faster':'slower'}</div><small>${Math.abs(c.reduction_pct).toFixed(1)}% ${faster?'reduction':'increase'} in observed time to 18</small><ul>${m.intervention.changed_variables.map(v=>`<li>${esc(v)}</li>`).join('')}</ul><p>${esc(m.intervention.comparison_note)}</p><a href="${attemptLink(r)}">Inspect attempt ↗</a></article>`;}).join('');
}
function renderProgress(){
  const rows=results.attempts.filter(r=>r.events.length && r.metadata);
  if(!rows.length){$('#progress-plot').innerHTML='<div class="chart-empty">No timestamped positive verdicts available.</div>';return;}
  const window=$('#curve-window').value, max=window==='all'?Math.max(54,...rows.flatMap(r=>r.events.map(e=>e.elapsed_s))):Number(window);
  const w=1080,h=315,left=55,right=25,top=35,bottom=40;
  const x=t=>left+t/max*(w-left-right),y=n=>h-bottom-n/18*(h-top-bottom);
  let svg=`<svg class="progress-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="Distinct correct questions over official elapsed seconds"><text x="${left}" y="18">Verified correct questions</text>`;
  for(let i=0;i<=6;i++){svg+=`<line class="grid" x1="${left}" x2="${w-right}" y1="${y(i*3)}" y2="${y(i*3)}"/><text x="${left-12}" y="${y(i*3)+4}" text-anchor="end">${i*3}</text><text text-anchor="middle" x="${x(max*i/6)}" y="${h-10}">${Math.round(max*i/6)}s</text>`;}
  svg+=`<line class="floor-line" x1="${x(54)}" x2="${x(54)}" y1="${top}" y2="${h-bottom}"/><text class="floor-label" x="${x(54)+7}" y="${top+16}">54s floor</text>`;
  rows.forEach((r,i)=>{let d=`M ${x(0)} ${y(0)}`;r.events.slice(0,18).forEach((e,n)=>{if(e.elapsed_s<=max)d+=` H ${x(e.elapsed_s)} V ${y(n+1)}`;});d+=` H ${x(Math.min(max,r.settlement_s??r.events.at(-1).elapsed_s))}`;svg+=`<path d="${d}" fill="none" stroke="${colors[i%colors.length]}" stroke-width="2.5"><title>${esc(r.metadata.label)}</title></path>`;});
  $('#progress-plot').innerHTML=svg+'</svg>';
  $('#curve-legend').innerHTML=rows.map((r,i)=>`<span><i style="background:${colors[i%colors.length]}"></i>${esc(r.metadata.label)}${r.time_to_18_s==null?' · target unmet':''}</span>`).join('');
}
function controlsTable(rows){
  return rows.map(r=>{const m=r.metadata;if(!m)return `<tr><td><a href="${attemptLink(r)}">${esc(r.id)}</a></td><td colspan="6">Invalid evidence; see warning above.</td></tr>`;
    const hp=m.controls.hyperparameters,g=m.gpu,c=r.comparison;
    const envelope=g.memory_utilization==null?'Unrecorded':`${Math.round(g.memory_utilization*100)}%${g.configured_envelope_mib!=null?` · ${(g.configured_envelope_mib/1024).toFixed(0)} GiB`:''}`;
    const changed=c?.changed_controls.map(v=>`${v.variable}: ${JSON.stringify(v.before)} → ${JSON.stringify(v.after)}`).join('\n');
    return `<tr><td><a href="${attemptLink(r)}">${esc(m.label)}</a><small>${esc(r.id)}<br>${esc(m.model.id)}<br>${esc(m.model.quantization??'Quantization unrecorded')} · ${esc(m.model.activation_dtype??'dtype unrecorded')}</small></td><td>${esc(m.intervention.label)}<small>${esc(r.attempt_status)}</small></td><td><span class="result-time">${sec(r.time_to_18_s)}</span><small class="${r.time_to_18_s==null?'unmet':''}">${esc(r.status)}</small><small>Settlement: ${sec(r.settlement_s)}</small></td><td>${r.solved??'—'} / ${m.controls.question_indices.length||'—'}</td><td>${hp.parallelism??'—'} × ${hp.rollouts??'—'}<small>${hp.first_pass_max_tokens??'—'} first-pass tokens<br>${hp.max_attempts_per_question??'—'} requests / question</small></td><td>${envelope}<small>${esc(g.device??'Device unrecorded')}</small></td><td><details><summary>Inspect controls</summary><p>${esc(m.intervention.comparison_note)}</p>${r.error?`<p class="unmet">${esc(r.error)}</p>`:''}<small>Runner: ${esc(m.runner.module)}<br>${esc(m.runner.version)}<br>Source: ${esc(m.runner.git_commit)}<br>Context: ${m.controls.max_context_tokens??'—'} tokens<br>Seed: ${hp.seed??'—'} · T ${hp.temperature??'—'} · top-p ${hp.top_p??'—'}</small>${changed?`<h4>Changed recorded controls</h4><pre>${esc(changed)}</pre><h4>Matched recorded controls</h4><pre>${esc(c.matched_controls.join('\n'))}</pre>`:''}<h4>Full metadata</h4><pre>${esc(JSON.stringify(m,null,2))}</pre>${r.metadata_missing?'':`<a href="/api/attempts/${encodeURIComponent(r.id)}/files/metadata.json" target="_blank" rel="noopener">metadata.json ↗</a>`}</details></td></tr>`;
  }).join('');
}
async function refresh(){
  $('#refresh').disabled=true;$('#error').hidden=true;
  try{
    const response=await fetch('/api/results',{cache:'no-store'});if(!response.ok)throw new Error(`Unable to load results (${response.status})`);results=await response.json();
    const rows=results.attempts,ranked=rows.filter(r=>r.time_to_18_s!=null).sort((a,b)=>a.time_to_18_s-b.time_to_18_s),best=ranked[0];
    $('#inventory').textContent=`${rows.length} saved attempts · ${ranked.length} measured targets reached`;
    $('#notice').textContent=results.warnings.join(' · ');$('#notice').hidden=!results.warnings.length;
    const metrics=[['Fastest observed',best?sec(best.time_to_18_s):'—',best?.metadata.label??'No measured target'],['Grader floor','54.00s','3s × 18 correct questions'],['Above the floor',best?sec(best.time_to_18_s-results.reference_floor_s):'—','Fastest run, after warmup']];
    $('#headline-metrics').innerHTML=metrics.map(([title,value,note],i)=>`<article class="metric-card ${i===0?'score-card':''}"><div class="metric-label">${title}</div><div class="metric-value">${value}</div><div class="metric-sub">${esc(note)}</div></article>`).join('');
    const history=attemptHistory(rows);
    $('#comparison-plot').innerHTML=comparisonPlot(history);$('#history-legend').innerHTML=historyLegend(history);$('#floor-note').textContent=results.floor_note;
    $('#excluded-note').textContent=rows.filter(r=>r.time_to_18_s==null || !Number.isFinite(Date.parse(r.attempt_started_at_utc??r.started_at_utc))).map(r=>`${r.metadata?.label??r.id}: ${r.time_to_18_s==null?r.status:'start timestamp unavailable'}${r.solved!=null?` (${r.solved} correct)`:''}`).join(' · ');
    $('#interventions').innerHTML=interventionCards(rows);$('#attempt-rows').innerHTML=controlsTable(rows);renderProgress();$('#content').hidden=false;
  }catch(e){$('#error').textContent=e.message;$('#error').hidden=false;}finally{$('#refresh').disabled=false;}
}
document.addEventListener('DOMContentLoaded',()=>{$('#refresh').addEventListener('click',refresh);$('#curve-window').addEventListener('change',renderProgress);refresh();});
