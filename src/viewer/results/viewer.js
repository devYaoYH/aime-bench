/**
 * Compare recorded time to 18 correct across canonical attempts. Use /results
 * through src.viewer_server after syncing evidence and annotating metadata.json.
 * Plot observed verdict times, show the serial grader floor, and expose changed
 * controls without treating single-run differences as isolated causal effects.
 */
const $ = selector => document.querySelector(selector);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const sec = v => v == null ? '—' : `${v.toFixed(2)}s`;
const colors = ['#147d65','#54729b','#b37b30','#865b89','#b65e46','#67804b'];
const attemptLink = row => `/?attempt=${encodeURIComponent(row.id)}`;
let results;
function comparisonPlot(rows) {
  if (!rows.length) return '<div class="chart-empty">No saved attempt has a measured time to 18 yet.</div>';
  const width=1080, left=305, right=85, top=65, rowHeight=78, height=top+rows.length*rowHeight+40;
  const max=Math.ceil(Math.max(60,...rows.map(r=>r.time_to_18_s))/60)*60;
  const x=t=>left+t/max*(width-left-right);
  let svg=`<svg class="comparison-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Measured time to eighteen correct answers with a 54 second grader floor">`;
  for(let i=0;i<=6;i++){const t=max*i/6;svg+=`<line class="grid" x1="${x(t)}" x2="${x(t)}" y1="${top-8}" y2="${height-35}"/><text text-anchor="middle" x="${x(t)}" y="${height-10}">${Math.round(t)}s</text>`;}
  const floor=results.reference_floor_s;
  svg+=`<line class="floor-line" x1="${x(floor)}" x2="${x(floor)}" y1="${top-20}" y2="${height-35}"/><text class="floor-label" x="${x(floor)+8}" y="25">54s grader floor · 18 × 3s</text>`;
  rows.forEach((r,i)=>{const y=top+i*rowHeight;const m=r.metadata;
    svg+=`<a href="${attemptLink(r)}"><text class="bar-label" x="0" y="${y+19}">${esc(m.label)}</text><text class="bar-note" x="0" y="${y+39}">${esc(m.intervention.label)}</text><rect x="${left}" y="${y+3}" width="${x(r.time_to_18_s)-left}" height="29" rx="4" fill="${colors[i%colors.length]}" opacity=".85"><title>${esc(m.label)}: ${sec(r.time_to_18_s)}</title></rect><text class="bar-value" x="${x(r.time_to_18_s)+10}" y="${y+24}">${sec(r.time_to_18_s)}</text></a>`;
  });
  return svg+'</svg>';
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
    $('#comparison-plot').innerHTML=comparisonPlot(ranked);$('#floor-note').textContent=results.floor_note;
    $('#excluded-note').textContent=rows.filter(r=>r.time_to_18_s==null).map(r=>`${r.metadata?.label??r.id}: ${r.status}${r.solved!=null?` (${r.solved} correct)`:''}`).join(' · ');
    $('#interventions').innerHTML=interventionCards(rows);$('#attempt-rows').innerHTML=controlsTable(rows);renderProgress();$('#content').hidden=false;
  }catch(e){$('#error').textContent=e.message;$('#error').hidden=false;}finally{$('#refresh').disabled=false;}
}
document.addEventListener('DOMContentLoaded',()=>{$('#refresh').addEventListener('click',refresh);$('#curve-window').addEventListener('change',renderProgress);refresh();});
