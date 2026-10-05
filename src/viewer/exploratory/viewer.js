/**
 * Render the original AIME run: sample votes, Jev scores,
 * timelines, and raw/final/reasoning tabs. Use through src.viewer_server when
 * inspecting run 20260930-155212; it reads the server's APIs and makes no
 * model requests. Detailed tabs require locally available raw traces.
 */
const $ = (selector) => document.querySelector(selector);
const VIEWER_RUN = '20260930-155212';
const state = { overview: null, selected: null, sample: 1, filter: 'all', search: '', activeTab: 'final', trace: null, requestId: 0 };
const integer = (value) => value == null ? '—' : new Intl.NumberFormat('en-US').format(value);
const seconds = (value, digits = 1) => value == null ? '—' : `${Number(value).toFixed(digits)}s`;
const minutes = (value) => value == null ? '—' : `${Math.floor(value / 60)}m ${String(Math.round(value % 60)).padStart(2, '0')}s`;
const percent = (value) => value == null ? '—' : `${Math.round(Number(value) * 100)}%`;
const elapsedSince = (time, start) => time && start ? (new Date(time) - new Date(start)) / 1000 : null;

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
}

function statusOf(item) {
  if (!item.answer_parseable) return 'missing';
  return item.correct ? 'correct' : 'wrong';
}

function showError(message) {
  const banner = $('#error');
  banner.textContent = message;
  banner.hidden = false;
}

async function getJson(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${response.status} while loading ${url}`);
  return response.json();
}

async function init() {
  try {
    $('#run-name').textContent = VIEWER_RUN;
    const params = new URLSearchParams(location.search);
    $('#search').addEventListener('input', event => { state.search = event.target.value.trim().toLowerCase(); renderQuestions(); });
    $('#filters').addEventListener('click', event => {
      const button = event.target.closest('[data-filter]');
      if (!button) return;
      state.filter = button.dataset.filter;
      document.querySelectorAll('.filter').forEach(node => node.classList.toggle('active', node === button));
      renderQuestions();
    });
    await loadRun(Number(params.get('q')) || 1, Number(params.get('attempt')) || 1);
  } catch (error) {
    showError(`Could not open viewer: ${error.message}`);
  }
}

async function loadRun(question = 1, sample = 1) {
  $('#error').hidden = true;
  $('#detail-panel').innerHTML = '<div class="loading">Loading run…</div>';
  try {
    state.overview = await getJson(`/api/runs/${VIEWER_RUN}/overview`);
    state.selected = null;
    state.trace = null;
    renderOverview();
    renderTimeline();
    renderReview();
    renderCalibration();
    renderAnalysis();
    renderQuestions();
    const available = state.overview.questions.find(item => item.problem_idx === question);
    await selectQuestion(available ? question : state.overview.questions[0].problem_idx, sample);
  } catch (error) {
    showError(`Could not load run ${VIEWER_RUN}: ${error.message}`);
  }
}

function renderOverview() {
  const { summary, config, questions } = state.overview;
  $('#model-name').textContent = summary.model;
  $('#run-date').textContent = new Date(summary.first_inference_request_at_utc).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
  $('#run-config').textContent = `${config.concurrency} concurrent · ${integer(config.max_tokens)} token cap · thinking on`;
  $('#pass-value').textContent = `${Math.round(summary.pass_at_1 * 100)}%`;
  $('#pass-detail').textContent = `${summary.correct} / ${summary.questions} correct`;
  $('#score-fill').style.width = `${summary.pass_at_1 * 100}%`;
  $('#wall-value').textContent = minutes(summary.wall_clock_first_request_to_all_grading_s);
  $('#wall-detail').textContent = `${Number(summary.wall_clock_first_request_to_all_grading_s).toFixed(3)}s · first request → final grade`;
  $('#latency-value').textContent = seconds(summary.api_latency_median_s, 0);
  $('#tokens-value').textContent = `${(summary.completion_tokens / 1000).toFixed(1)}k`;
  $('#tokens-detail').textContent = `${integer(summary.prompt_tokens)} input · ${integer(summary.reasoning_tokens_reported)} reasoning`;
  $('#format-value').textContent = `${questions.filter(item => !item.format_valid).length} / ${questions.length}`;
  const expansion = state.overview.self_consistency;
  document.querySelectorAll('.expansion-metric').forEach(node => { node.hidden = !expansion; });
  $('#overview-subtitle').textContent = expansion
    ? 'Original pass@1 run plus seven more independent samples per question; all eight are grouped below.'
    : 'One generated response per question, scored against the AIME answer key.';
  $('#timeline-note').textContent = expansion
    ? 'This timeline shows the original pass@1 requests. Select a question to inspect all eight attempts and their individual latencies.'
    : "Bars show each question’s API request span on the shared run clock. Select a bar to inspect its response.";
  if (expansion) {
    $('#pass8-value').textContent = percent(expansion.pass_at_8);
    $('#pass8-detail').textContent = `${expansion.pass_at_8_correct} / ${expansion.questions} had at least one correct answer`;
    $('#pass8-fill').style.width = percent(expansion.pass_at_8);
    $('#vote-value').textContent = percent(expansion.majority_vote_accuracy);
    $('#vote-detail').textContent = `${expansion.majority_vote_correct} / ${expansion.questions} correct · ${expansion.questions_with_strict_majority} reached a majority · ${expansion.modal_vote_correct} modal votes correct`;
    $('#expansion-wall-value').textContent = minutes(expansion.wall_clock_first_request_to_all_grading_s);
    $('#expansion-wall-detail').textContent = `${Number(expansion.wall_clock_first_request_to_all_grading_s).toFixed(3)}s · ${expansion.configured_concurrency ?? 30} concurrent · request → grade`;
  }
}

function renderTimeline() {
  const { summary, questions } = state.overview;
  const wall = summary.wall_clock_first_request_to_all_grading_s;
  $('#timeline-axis').innerHTML = [0, .25, .5, .75, 1].map(fraction => `<span>${minutes(wall * fraction)}</span>`).join('');
  const host = $('#timeline');
  host.replaceChildren();
  for (const item of questions) {
    const row = document.createElement('button');
    row.type = 'button';
    row.className = `timeline-row ${statusOf(item) === 'missing' ? 'format' : statusOf(item)}${state.selected === item.problem_idx ? ' selected' : ''}`;
    row.title = `Q${item.problem_idx}: ${seconds(item.api_latency_s)} API · ${integer(item.completion_tokens)} output tokens`;
    row.setAttribute('aria-label', row.title);
    const label = document.createElement('span');
    label.className = 'timeline-q';
    label.textContent = String(item.problem_idx).padStart(2, '0');
    const track = document.createElement('span');
    track.className = 'timeline-track';
    const bar = document.createElement('span');
    bar.className = 'timeline-bar';
    const start = Math.max(0, item.request_start_s ?? 0);
    const length = Math.max(0, (item.request_end_s ?? start) - start);
    bar.style.left = `${Math.min(100, start / wall * 100)}%`;
    bar.style.width = `${Math.min(100 - start / wall * 100, length / wall * 100)}%`;
    track.append(bar);
    row.append(label, track);
    row.addEventListener('click', () => selectQuestion(item.problem_idx));
    host.append(row);
  }
}

function renderReview() {
  const review = state.overview?.jev_review;
  const card = $('#review-card');
  card.hidden = !review;
  if (!review) return;
  $('#review-meta').textContent = `${review.judged} / ${review.failed_trajectories} failed traces reviewed`;
  const prefixByIndex = new Map((state.overview.prefix_review?.ranked || []).map(row => [row.problem_idx, row]));
  const host = $('#review-ranking');
  host.replaceChildren();
  for (const row of review.ranked) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `review-row${state.selected === row.problem_idx ? ' selected' : ''}`;
    const prefix = prefixByIndex.get(row.problem_idx);
    button.setAttribute('aria-label', `Question ${row.problem_idx}, ${percent(row.promising_to_extend)} full trace promising, ${percent(prefix?.prefix_promising)} prefix promising, ${percent(row.near_solution)} near a solution, ${percent(row.stuck_or_repeating)} stuck or repeating`);
    button.innerHTML = `<span class="review-q">Q${String(row.problem_idx).padStart(2, '0')}</span><span class="review-prospect"><span class="review-bar"><span style="width:${Math.max(0, Math.min(100, row.promising_to_extend * 100))}%"></span></span><strong>${percent(row.promising_to_extend)}</strong></span><span class="review-signal">${percent(prefix?.prefix_promising)}</span><span class="review-signal">${percent(row.near_solution)}</span><span class="review-signal">${percent(row.stuck_or_repeating)}</span>`;
    button.addEventListener('click', () => {
      selectQuestion(row.problem_idx);
      $('#detail-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    host.append(button);
  }
}

function calibrationFor(index, sample) {
  return state.overview?.jev_calibration?.rows.find(row => row.problem_idx === index && row.sample_number === sample)
    ?? state.overview?.jev_calibration_followup?.rows.find(row => row.problem_idx === index && row.sample_number === sample);
}

function renderCalibration() {
  const calibration = state.overview?.jev_calibration;
  const card = $('#calibration-card');
  card.hidden = !calibration;
  if (!calibration) return;
  const followup = state.overview?.jev_calibration_followup;
  const total = calibration.completed_trajectories + (followup?.new_scored ?? 0);
  $('#calibration-meta').textContent = `${total} / 240 attempts scored`;
  $('#calibration-coverage').textContent = followup
    ? `${total} trajectories assessed from their first 1,500 reasoning tokens. A missing final answer is an observed failure under the original 16K cap, not proof that a prefix is unsalvageable.`
    : `${calibration.completed_trajectories} completed responses were scored; ${240-total} capped responses were excluded and remain unscored. Missing labels mean not assessed, not rejected. This selected cohort cannot establish overall precision.`;
  const retained = followup?.thresholds.find(row => row.retain_at_or_above === 0.5);
  $('#calibration-followup').innerHTML = retained
    ? `<div class="calibration-stat"><span>No-answer traces retained · score ≥ 50%</span><strong>${retained.capped_retained} / ${followup.new_scored}</strong><small>capped traces assessed</small></div><div class="calibration-stat"><span>Observed final-answer precision · score ≥ 50%</span><strong>${(retained.observed_final_precision*100).toFixed(1)}%</strong><small>${retained.correct_final_retained} / ${retained.retained} retained traces returned a correct final answer under the original cap; ${retained.completed_wrong_retained} completed wrong answers also retained</small></div>`
    : '';
  const atHalf = calibration.thresholds.find(row => row.reject_below === 0.5);
  $('#calibration-stats').innerHTML = `<div class="calibration-stat"><span>Missed within budget · reject below 50%</span><strong>${atHalf.strict_budget_false_negatives} / ${atHalf.strict_budget_positives}</strong><small>correct completions by 9,692 total output tokens</small></div><div class="calibration-stat"><span>Eventually correct · reject below 50%</span><strong>${atHalf.eventual_false_negatives} / ${atHalf.eventual_positives}</strong><small>includes answers after the stated continuation budget</small></div>`;
  $('#calibration-thresholds').innerHTML = `<div class="calibration-threshold-head"><span>Reject below</span><span>Missed within budget</span><span>Missed eventually correct</span></div>${calibration.thresholds.filter(row => [0.5, 0.6, 0.65, 0.7].includes(row.reject_below)).map(row => `<div class="calibration-threshold-row"><strong>${percent(row.reject_below)}</strong><span>${row.strict_budget_false_negatives} / ${row.strict_budget_positives}</span><span>${row.eventual_false_negatives} / ${row.eventual_positives}</span></div>`).join('')}`;
}

function renderAnalysis() {
  const checkpoints = state.overview?.reasoning_checkpoints;
  const budget = state.overview?.budget_analysis;
  const card = $('#analysis-card');
  card.hidden = !checkpoints || !budget;
  if (!checkpoints || !budget) return;
  $('#analysis-meta').textContent = '240 saved trajectories · 30 questions';
  const plotUrl = `/api/runs/${encodeURIComponent(state.overview.run_id)}/reasoning-tokens.svg`;
  $('#analysis-plot-link').href = plotUrl;
  $('#analysis-plot').src = plotUrl;
  const checkpointRows = checkpoints.checkpoints.filter(row => [1500, 5000, 9000, 13000, 15000].includes(row.reasoning_checkpoint_tokens));
  $('#analysis-checkpoints').innerHTML = `<div class="analysis-table-head"><span>Reasoning checkpoint</span><span>Still running</span><span>Correct in next 4k output</span><span>Correct by original cap</span></div>${checkpointRows.map(row => { const next = row.windows.find(window => window.additional_output_tokens === 4096); return `<div class="analysis-table-row"><strong>${(row.reasoning_checkpoint_tokens / 1000).toFixed(1)}k</strong><span>${row.active_trajectories} / 240</span><span>${next.correct_final_answers} / ${row.active_trajectories} · ${percent(next.rate)}</span><span>${row.eventually_correct_by_original_cap} / ${row.active_trajectories} · ${percent(row.eventually_correct_by_original_cap / row.active_trajectories)}</span></div>`; }).join('')}`;
  const frontierRows = budget.generation_pass_at_n.filter(row => [1, 2, 3, 4, 5, 8].includes(row.generations_per_question));
  $('#analysis-frontier').innerHTML = `<div class="analysis-frontier-grid">${frontierRows.map(row => `<div><span>${row.generations_per_question} generation${row.generations_per_question === 1 ? '' : 's'} / question</span><strong>${row.correct_questions} / 30</strong><small>${row.model_requests} model calls · ${row.final_answers_generated} final responses</small></div>`).join('')}</div><p>In a retrospective stop-on-first-final replay, ${budget.stop_on_first_final.model_requests} model calls produced ${budget.stop_on_first_final.questions_with_final_candidate} final candidates and the same ${budget.stop_on_first_final.correct_questions}/30 correct questions. Sequential retries may take longer on the slowest question, so this is a request-count result, not a measured wall-clock improvement.</p>`;
}

function visibleQuestions() {
  const { questions } = state.overview;
  return questions.filter(item => {
    const matchesFilter = state.filter === 'all' || (state.filter === 'format' ? !item.format_valid : statusOf(item) === state.filter);
    const matchesSearch = !state.search || String(item.problem_idx).includes(state.search) || item.problem.toLowerCase().includes(state.search);
    return matchesFilter && matchesSearch;
  });
}

function renderQuestions() {
  if (!state.overview) return;
  const questions = visibleQuestions();
  $('#visible-count').textContent = questions.length;
  const host = $('#question-list');
  host.replaceChildren();
  if (!questions.length) {
    host.innerHTML = '<div class="loading">No questions match this view.</div>';
    return;
  }
  for (const item of questions) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `question-item${state.selected === item.problem_idx ? ' selected' : ''}`;
    button.setAttribute('aria-label', `Question ${item.problem_idx}, ${statusOf(item)}, ${seconds(item.api_latency_s)} API latency`);
    const num = document.createElement('span');
    num.className = 'question-num';
    num.textContent = String(item.problem_idx).padStart(2, '0');
    const excerpt = document.createElement('span');
    excerpt.className = 'question-text';
    excerpt.textContent = item.problem;
    const meta = document.createElement('span');
    meta.className = 'question-meta';
    const dot = document.createElement('i');
    dot.className = `dot ${statusOf(item) === 'correct' ? 'good' : statusOf(item) === 'missing' ? 'bad' : 'wrong'}`;
    meta.append(dot, document.createTextNode(item.self_consistency ? `${item.self_consistency.correct_attempts}/8 · ${seconds(item.api_latency_s, 0)}` : seconds(item.api_latency_s, 0)));
    button.append(num, excerpt, meta);
    button.addEventListener('click', () => selectQuestion(item.problem_idx));
    host.append(button);
  }
}

async function selectQuestion(index, sample = 1, preserveTab = false) {
  if (!Number.isInteger(sample) || sample < 1 || sample > 8 || (sample > 1 && !state.overview?.self_consistency)) sample = 1;
  const requestId = ++state.requestId;
  state.selected = index;
  state.sample = sample;
  state.trace = null;
  if (!preserveTab) state.activeTab = 'final';
  renderTimeline();
  renderReview();
  renderQuestions();
  $('#detail-panel').innerHTML = '<div class="loading">Loading response trace…</div>';
  const runId = state.overview.run_id;
  const url = new URL(location.href);
  url.searchParams.set('run', runId);
  url.searchParams.set('q', index);
  if (sample === 1) url.searchParams.delete('attempt');
  else url.searchParams.set('attempt', sample);
  history.replaceState(null, '', url);
  try {
    const trace = await getJson(sample === 1
      ? `/api/runs/${encodeURIComponent(runId)}/questions/${index}`
      : `/api/runs/${encodeURIComponent(runId)}/attempts/${index}/${sample}`);
    if (requestId !== state.requestId) return;
    state.trace = trace;
    renderDetail();
  } catch (error) {
    if (requestId === state.requestId) showError(`Could not load question ${index}: ${error.message}`);
  }
}

function renderDetail() {
  const trace = state.trace;
  const item = state.overview.questions.find(row => row.problem_idx === trace.problem_idx);
  const message = trace.response?.choices?.[0]?.message || {};
  const final = message.content || '';
  const reasoning = message.reasoning || '';
  const finalLine = final.trim().split('\n').at(-1)?.trim() || '';
  const formatValid = /^Answer:\s*\d{1,3}$/.test(finalLine);
  const status = trace.candidate == null ? 'missing' : trace.correct ? 'correct' : 'wrong';
  const statusText = status === 'correct' ? 'Correct' : status === 'missing' ? 'No final answer' : 'Wrong answer';
  const cap = state.overview.config.max_tokens;
  const usage = trace.usage || {};
  const reasoningTokens = usage.completion_tokens_details?.reasoning_tokens;
  const expansion = state.overview.self_consistency;
  const end = state.sample === 1 ? item.grading_end_s : elapsedSince(trace.grading_completed_at_utc, expansion?.first_inference_request_at_utc);
  const clockLabel = state.sample === 1 ? 'original run' : 'seven-sample run';
  const group = item.self_consistency;
  const votes = group?.votes.map(row => `${row.answer} × ${row.count}`).join(' · ') || 'No parseable answers';
  const calibration = calibrationFor(item.problem_idx, state.sample);
  const groupHtml = group ? `<section class="attempt-group" aria-label="Eight attempts for question ${item.problem_idx}">
      <div class="attempt-group-head"><div><div class="eyebrow">Self-consistency / question ${String(item.problem_idx).padStart(2, '0')}</div><h3>Eight independent attempts</h3></div><span class="badge ${group.pass_at_8 ? 'good' : 'bad'}">Pass@8 ${group.pass_at_8 ? 'yes' : 'no'} · ${group.correct_attempts}/8 correct</span></div>
      <div class="vote-summary"><span>Modal vote: <strong>${group.modal_answer == null ? 'No unique mode' : group.modal_answer}</strong>${group.modal_answer == null ? '' : ` (${group.modal_votes}/8)`}</span><span>Strict majority: <strong>${group.majority_answer == null ? 'None' : group.majority_answer}</strong></span><span>Answers: ${escapeHtml(votes)}</span></div>
      <div id="attempt-grid" class="attempt-grid">${group.attempts.map(row => { const judged = calibrationFor(item.problem_idx, row.attempt); return `<button type="button" class="attempt-button ${row.correct ? 'correct' : row.candidate == null ? 'missing' : 'wrong'}${state.sample === row.attempt ? ' selected' : ''}" data-attempt="${row.attempt}" aria-label="Attempt ${row.attempt}, ${row.candidate == null ? 'no final answer' : `answer ${escapeHtml(row.candidate)}`}, ${row.correct ? 'correct' : 'incorrect'}${judged ? `, Jev promising ${percent(judged.probability)}` : ', Jev not scored'}"><span>Attempt ${row.attempt}${row.attempt === 1 ? ' · original' : ''}</span><strong>${row.candidate == null ? 'No answer' : escapeHtml(row.candidate)}</strong><small>${seconds(row.api_latency_s, 0)} · ${integer(row.completion_tokens)} tok</small><small class="attempt-jev">${judged ? `Jev promising ${percent(judged.probability)}` : 'Jev not scored'}</small></button>`; }).join('')}</div>
    </section>` : '';
  const review = item.jev_review;
  const prefix = item.prefix_review;
  const reviewHtml = review ? `<section class="review-detail" aria-label="Jev continuation review">
      <div class="review-detail-head"><div><div class="eyebrow">Jev 1.13 / Noul decisions · original attempt</div><h3>Continuation outlook</h3></div><div class="review-links"><a class="trace-link" href="/api/runs/${encodeURIComponent(state.overview.run_id)}/jev-review/${item.problem_idx}" target="_blank" rel="noopener">Full trace decision ↗</a>${prefix ? `<a class="trace-link" href="/api/runs/${encodeURIComponent(state.overview.run_id)}/jev-prefix/${item.problem_idx}" target="_blank" rel="noopener">Prefix decision ↗</a>` : ''}</div></div>
      <div class="review-detail-grid"><div><span>Full trace promising</span><strong>${percent(review.promising_to_extend)}</strong></div><div><span>First 1,500 tokens</span><strong>${percent(prefix?.prefix_promising)}</strong></div><div><span>Coherent route</span><strong>${percent(review.coherent_route)}</strong></div><div><span>Near solution</span><strong>${percent(review.near_solution)}</strong></div><div><span>Stuck / repeating</span><strong>${percent(review.stuck_or_repeating)}</strong></div></div>
      <p>Jev reviewed attempt 1 with no answer key. Its estimates are model judgments, not observed continuation outcomes.</p>
    </section>` : '';
  const afterPrefix = calibration ? calibration.completion_tokens - 1500 : null;
  const calibrationOutcome = calibration?.finish_reason === 'length'
    ? `It hit the original ${integer(calibration.completion_tokens)}-output-token cap without a final answer. This observed failure does not establish that the prefix was unsalvageable.`
    : calibration?.correct
    ? calibration.within_8192_more_tokens
      ? `It reached the correct final answer after ${integer(afterPrefix)} more output tokens, within the stated budget.`
      : `It reached the correct final answer after ${integer(afterPrefix)} more output tokens, beyond the stated 8,192-token budget.`
    : `It completed with an incorrect final answer after ${integer(afterPrefix)} more output tokens. This one continuation does not establish that the prefix was unsalvageable.`;
  const calibrationHtml = calibration ? `<section class="calibration-detail" aria-label="Jev prefix decision compared with observed completion"><div class="review-detail-head"><div><div class="eyebrow">Jev 1.13 / first 1,500 reasoning tokens</div><h3>Prediction versus observed continuation</h3></div><a class="trace-link" href="/api/runs/${encodeURIComponent(state.overview.run_id)}/${calibration.cohort === 'post-cutoff capped follow-up' ? 'jev-calibration-followup' : 'jev-calibration'}/${item.problem_idx}/${state.sample}" target="_blank" rel="noopener">Decision JSON ↗</a></div><div class="calibration-detail-main"><strong>${percent(calibration.probability)}</strong><span>Jev “promising to extend”</span><span class="badge ${calibration.probability < 0.5 && calibration.correct ? 'bad' : 'neutral'}">${calibration.probability < 0.5 ? 'Below 50% cutoff' : 'At or above 50%'}</span></div><p>${calibrationOutcome} Jev saw neither this result nor the answer key.</p></section>` : '';
  const formatExplanation = formatValid
    ? 'The final line follows the requested Answer: NNN format.'
    : trace.candidate != null
      ? 'An integer was extracted, but the final line does not follow the requested Answer: NNN format.'
      : trace.finish_reason === 'length'
        ? 'The output hit its token cap without a final answer or the requested Answer: NNN line.'
        : 'No final answer or requested Answer: NNN line was returned.';
  const traceUrl = state.sample === 1
    ? `/api/runs/${encodeURIComponent(state.overview.run_id)}/questions/${item.problem_idx}`
    : `/api/runs/${encodeURIComponent(state.overview.run_id)}/attempts/${item.problem_idx}/${state.sample}`;
  $('#detail-panel').innerHTML = `
    ${groupHtml}
    <div class="detail-top">
      <div class="detail-header"><div><div class="eyebrow">Response inspection / ${String(item.problem_idx).padStart(2, '0')} / attempt ${state.sample}</div><h2>Question ${item.problem_idx}</h2></div>
      <div class="detail-actions"><span class="badge ${status === 'correct' ? 'good' : status === 'missing' ? 'bad' : 'wrong'}">${statusText}</span><span class="badge ${formatValid ? 'good' : 'bad'}">${formatValid ? 'Format valid' : 'Format incorrect'}</span><a class="trace-link" href="${traceUrl}" target="_blank" rel="noopener">Full JSON ↗</a></div></div>
      <div class="answer-row"><div class="answer-box ${trace.candidate == null ? 'missing' : ''}"><div class="small-label">Model answer</div><strong>${trace.candidate == null ? 'Missing' : escapeHtml(trace.candidate)}</strong></div><div class="answer-box"><div class="small-label">Official answer</div><strong>${escapeHtml(trace.gold_answer)}</strong></div><div class="answer-explanation">${escapeHtml(formatExplanation)}</div></div>
    </div>
    <div class="detail-stats">
      <div class="detail-stat"><div class="small-label">API latency</div><strong>${seconds(trace.total_api_latency_s)}</strong><small>${trace.api_attempts?.length ?? 0} request${trace.api_attempts?.length === 1 ? '' : 's'}</small></div>
      <div class="detail-stat"><div class="small-label">Output length</div><strong>${integer(usage.completion_tokens)} tok</strong><small>${cap ? Math.round((usage.completion_tokens || 0) / cap * 100) : 0}% of cap</small></div>
      <div class="detail-stat"><div class="small-label">Input / reasoning</div><strong>${integer(usage.prompt_tokens)} / ${integer(reasoningTokens)}</strong><small>provider tokens</small></div>
      <div class="detail-stat"><div class="small-label">Response text</div><strong>${integer(final.length)} chars</strong><small>${integer(reasoning.length)} reasoning chars</small></div>
      <div class="detail-stat"><div class="small-label">Run clock</div><strong>+${seconds(end)}</strong><small>graded in ${clockLabel}</small></div>
    </div>
    ${reviewHtml}
    ${calibrationHtml}
    <section class="content-section"><h3>Problem statement</h3><pre class="problem-text"></pre></section>
    <div class="response-heading"><div><div class="eyebrow">Full generation output · attempt ${state.sample}</div><h3>Model response</h3></div><span class="badge neutral">Finish: ${escapeHtml(trace.finish_reason ?? 'unknown')}</span></div>
    <div class="response-tabs" role="tablist" aria-label="Response content"><button class="response-tab ${state.activeTab === 'final' ? 'active' : ''}" data-tab="final" role="tab" type="button">Final response</button><button class="response-tab ${state.activeTab === 'reasoning' ? 'active' : ''}" data-tab="reasoning" role="tab" type="button">Reasoning trace</button><button class="response-tab ${state.activeTab === 'raw' ? 'active' : ''}" data-tab="raw" role="tab" type="button">Raw response JSON</button></div>
    <div id="response-body" class="response-body" role="tabpanel"></div>`;
  $('#detail-panel .problem-text').textContent = item.problem;
  $('#attempt-grid')?.addEventListener('click', event => {
    const button = event.target.closest('[data-attempt]');
    if (button) selectQuestion(item.problem_idx, Number(button.dataset.attempt), true);
  });
  $('#detail-panel .response-tabs').addEventListener('click', event => {
    const button = event.target.closest('[data-tab]');
    if (!button) return;
    state.activeTab = button.dataset.tab;
    document.querySelectorAll('.response-tab').forEach(node => node.classList.toggle('active', node === button));
    renderResponseBody();
  });
  renderResponseBody();
}

function renderResponseBody() {
  const host = $('#response-body');
  host.replaceChildren();
  const trace = state.trace;
  const message = trace.response?.choices?.[0]?.message || {};
  let value = state.activeTab === 'reasoning' ? message.reasoning : state.activeTab === 'raw' ? JSON.stringify(trace.response, null, 2) : message.content;
  if (!value) {
    const placeholder = document.createElement('div');
    placeholder.className = 'response-placeholder';
    placeholder.textContent = state.activeTab === 'final' ? 'No final response text was returned. Inspect the reasoning trace for the generated output.' : 'No content was returned for this view.';
    host.append(placeholder);
    return;
  }
  const pre = document.createElement('pre');
  pre.className = state.activeTab === 'raw' ? 'raw' : '';
  pre.textContent = value;
  host.append(pre);
}

document.addEventListener('DOMContentLoaded', init);
