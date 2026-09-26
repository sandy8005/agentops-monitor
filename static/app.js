var NL = String.fromCharCode(10);
    var pollTimer = null;
    var _csrfToken = null;

    // The CSRF token is bound to the session and returned by GET /csrf in the JSON
    // body (there is no readable cookie). Fetch once and cache; it's cleared on
    // logout so the next session gets a fresh token.
    async function ensureCsrf() {
      if (_csrfToken) return _csrfToken;
      try {
        const r = await fetch('/csrf');
        const j = await r.json();
        _csrfToken = (j && j.csrf_token) || null;
      } catch (e) { /* leave null; state-changing calls will 403 and prompt reload */ }
      return _csrfToken;
    }

    // fetch() wrapper that attaches the CSRF header on state-changing requests.
    async function csrfFetch(url, opts) {
      opts = opts || {};
      const method = (opts.method || 'GET').toUpperCase();
      if (method !== 'GET' && method !== 'HEAD') {
        await ensureCsrf();
        opts.headers = Object.assign({}, opts.headers, { 'X-CSRF-Token': _csrfToken || '' });
      }
      return fetch(url, opts);
    }

    // Run lifecycle vocabulary (mirrors the backend). ACTIVE runs keep the detail
    // view polling — a run that is 'queued' or 'retrying' is still going to change,
    // it just hasn't been picked up by a worker yet.
    var ACTIVE_STATUSES = ['queued', 'retrying', 'running'];
    function isActive(status) { return ACTIVE_STATUSES.indexOf(status) !== -1; }

    // Every status gets explicit styling (falls back to 'errors' only for unknowns).
    var STATUS_CLASS = {
      success: 'success', failed: 'failed', running: 'running', queued: 'queued',
      retrying: 'retrying', waiting_for_human: 'waiting', cancelled: 'cancelled',
      no_matches: 'no_matches', completed_with_errors: 'errors'
    };
    function statusClass(status) { return STATUS_CLASS[status] || 'errors'; }

    // Score breakdown. New rows store {earned, max} per category — the max is the
    // REAL renormalized weight (e.g. required is worth 62.5 when the job lists no
    // preferred skills). Legacy rows are flat numbers; for those we show the earned
    // points only rather than inventing a fixed /50 /20 /15 /15 denominator.
    function renderBreakdown(b) {
      var parts = [];
      ['required', 'preferred', 'projects', 'experience'].forEach(function(cat) {
        var v = b[cat];
        if (v === undefined || v === null) return;
        if (typeof v === 'object') {
          if (!v.max) { parts.push(cat + ' n/a'); return; }
          parts.push(cat + ' ' + escapeHtml(v.earned) + '/' + escapeHtml(v.max));
        } else {
          parts.push(cat + ' ' + escapeHtml(v));
        }
      });
      return parts.join(' &nbsp; ');
    }

    function escapeHtml(s) {
      return String(s == null ? '' : s)
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
    }

    async function uploadResume() {
      const fileInput = document.getElementById('resumeFile');
      const msg = document.getElementById('uploadMsg');
      msg.textContent = '';
      if (!fileInput.files.length) { msg.textContent = 'Please choose a PDF file.'; return; }
      const file = fileInput.files[0];
      if (file.size > 5 * 1024 * 1024) { msg.textContent = 'File too large (max 5 MB).'; return; }

      const btn = document.getElementById('uploadBtn');
      btn.disabled = true; btn.textContent = 'Uploading...';
      msg.textContent = 'Extracting and storing resume...';

      const fd = new FormData();
      fd.append('file', file);
      fd.append('name', document.getElementById('resumeName').value.trim());

      try {
        const up = await csrfFetch('/upload', { method: 'POST', body: fd });
        if (!up.ok) { const e = await up.json(); throw new Error(e.detail || 'upload failed'); }
        const upData = await up.json();
        msg.textContent = 'Stored resume #' + upData.resume_id + ' (' + upData.chars + ' chars). Set search options below and run it.';
        loadResumes();
      } catch (err) {
        msg.textContent = 'Error: ' + err.message;
      } finally {
        btn.disabled = false; btn.innerHTML = '&#9654; Upload Resume';
      }
    }

    async function loadResumes() {
      const div = document.getElementById('resumeLibrary');
      try {
        const res = await fetch('/resumes');
        if (!res.ok) throw new Error('failed to load resumes');
        const resumes = await res.json();
        if (resumes.length === 0) { div.innerHTML = '<div class="step">No saved resumes yet. Upload one above.</div>'; return; }
        div.innerHTML = '';
        for (const r of resumes) {
          const el = document.createElement('div');
          el.className = 'step';
          el.innerHTML = '<strong>' + escapeHtml(r.name) + '</strong> <span style="color:#8b8f9c;">(' + escapeHtml(r.chars) + ' chars)</span>' +
            '<div class="search-form">' +
              '<div class="row">' +
                '<div><label>Target role</label><br><input type="text" id="role_' + r.id + '" placeholder="AI/ML Engineer" style="width:95%;"></div>' +
                '<div><label>Location <span style="color:#8b8f9c;">(filters Adzuna; remote-only sources ignore it)</span></label><br><input type="text" id="loc_' + r.id + '" placeholder="Michigan" style="width:95%;"></div>' +
              '</div>' +
              '<div class="row" style="margin-top:6px;">' +
                '<div><label>Work mode</label><br><select id="mode_' + r.id + '" style="width:100%;">' +
                  '<option value="">(any)</option><option value="remote">remote</option><option value="hybrid">hybrid</option><option value="onsite">onsite</option>' +
                '</select></div>' +
                '<div><label>Employment type</label><br><select id="emp_' + r.id + '" style="width:100%;">' +
                  '<option value="">(any)</option><option value="full-time">full-time</option><option value="part-time">part-time</option><option value="contract">contract</option><option value="internship">internship</option>' +
                '</select></div>' +
              '</div>' +
              '<div style="margin-top:8px;">' +
                '<label style="color:#e4e6eb;"><input type="checkbox" id="eval_' + r.id + '" style="width:auto;margin-right:6px;">Run LLM evaluator (extra API calls)</label>' +
              '</div>' +
              '<div style="margin-top:8px;">' +
                '<button class="btn-sm btn-approve" onclick="runResume(' + r.id + ')">Run Search</button>' +
                '<button class="btn-sm btn-reject" onclick="deleteResume(' + r.id + ')">Delete</button>' +
              '</div>' +
            '</div>';
          div.appendChild(el);
        }
      } catch (err) {
        div.innerHTML = '<div class="step" style="color:#f87171;">Could not load resumes: ' + escapeHtml(err.message) + '</div>';
      }
    }

    async function runResume(id) {
      const role = (document.getElementById('role_' + id).value || '').trim();
      if (!role) { alert('Please enter a target role for this search.'); return; }
      const loc = (document.getElementById('loc_' + id).value || '').trim();
      const mode = document.getElementById('mode_' + id).value;
      const emp = document.getElementById('emp_' + id).value;
      const doEval = document.getElementById('eval_' + id).checked ? 'true' : 'false';
      const qs = '?resume_id=' + id +
                 '&target_role=' + encodeURIComponent(role) +
                 '&location=' + encodeURIComponent(loc) +
                 '&work_mode=' + encodeURIComponent(mode) +
                 '&employment_type=' + encodeURIComponent(emp) +
                 '&evaluate=' + doEval;
      try {
        const run = await csrfFetch('/runs' + qs, { method: 'POST' });
        if (!run.ok) { const e = await run.json(); throw new Error(e.detail || 'run failed'); }
        const runData = await run.json();
        setTimeout(function() { loadRuns(); loadDetail(runData.run_id); }, 1500);
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function cancelRun(id) {
      if (!confirm('Cancel run #' + id + '?')) return;
      try {
        const res = await csrfFetch('/runs/' + id + '/cancel', { method: 'POST' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'cancel failed'); }
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function deleteResume(id) {
      if (!confirm('Delete resume #' + id + '?')) return;
      try {
        const res = await csrfFetch('/resumes/' + id, { method: 'DELETE' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'delete failed'); }
        loadResumes();
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function loadRuns() {
      const tbody = document.querySelector('#runs tbody');
      try {
        const res = await fetch('/runs');
        if (!res.ok) throw new Error('failed to load runs');
        const runs = await res.json();
        tbody.innerHTML = '';
        for (const r of runs) {
          const cls = statusClass(r.status);
          const searchDesc = (r.target_role || '-') + (r.location ? ' / ' + r.location : '') + (r.work_mode ? ' / ' + r.work_mode : '');
          const tr = document.createElement('tr');
          tr.innerHTML = '<td>#' + escapeHtml(r.id) + '</td>' +
            '<td><span class="status ' + cls + '">' + escapeHtml(r.status) + '</span>' +
              (r.error_code ? ' <span class="code">' + escapeHtml(r.error_code) + '</span>' : '') + '</td>' +
            '<td style="font-size:12px;color:#b0b4c0;">' + escapeHtml(searchDesc) + '</td>' +
            '<td>' + escapeHtml((r.started_at || '').replace('T', ' ').slice(0, 16)) + '</td>' +
            '<td>' + escapeHtml(r.total_tokens || 0) + '</td>' +
            '<td>~$' + escapeHtml((r.total_cost || 0).toFixed(6)) + '</td>';
          tr.onclick = function() { loadDetail(r.id); };
          tbody.appendChild(tr);
        }
      } catch (err) {
        tbody.innerHTML = '<tr><td colspan="6" style="color:#f87171;">Could not load runs: ' + escapeHtml(err.message) + '</td></tr>';
      }
    }

    async function loadDetail(id) {
      const d = document.getElementById('detail');
      try {
        const res = await fetch('/runs/' + id);
        if (!res.ok) throw new Error('failed to load run ' + id);
        const run = await res.json();

        const total = run.steps.length;
        const done = run.steps.filter(function(s) {
          return s.status === 'success' || s.status === 'failed';
        }).length;
        const active = isActive(run.status);
        const running = run.status === 'running';

        let html = '<h2>Run #' + escapeHtml(run.id) + ' - <span class="status ' + statusClass(run.status) + '">' + escapeHtml(run.status) + '</span>';
        if (run.target_role) html += ' <span class="note">(' + escapeHtml(run.target_role) + (run.location ? ' / ' + escapeHtml(run.location) : '') + ')</span>';
        if (run.attempt && run.attempt > 1) html += ' <span class="retry">attempt ' + escapeHtml(run.attempt) + '</span>';
        html += '</h2>';

        // Machine-readable outcome — the reason the error taxonomy exists.
        if (run.error_code || run.stop_reason) {
          html += '<div class="step outcome ' + (run.status === 'failed' ? 'outcome-failed' : '') + '">' +
            '<strong>' + escapeHtml(String(run.status).toUpperCase()) + '</strong>' +
            (run.error_code ? '<div>Code: <span class="code">' + escapeHtml(String(run.error_code).toUpperCase()) + '</span></div>' : '') +
            (run.stop_reason ? '<div>Reason: ' + escapeHtml(run.stop_reason) + '</div>' : '') +
            '</div>';
        }

        if (active && !running) {
          html += '<div class="step progress"><strong class="' + statusClass(run.status) + '-text">&#9679; ' +
            escapeHtml(String(run.status).toUpperCase()) + '</strong> &nbsp; ' +
            (run.status === 'retrying' ? 'the last attempt failed with a transient error; waiting to retry'
                                       : 'waiting for a worker to pick this run up') +
            ' &nbsp; <button class="btn-sm btn-reject" onclick="cancelRun(' + run.id + ')">Cancel</button></div>';
        }

        if (running) {
          const active = run.steps.filter(function(s){ return s.status === 'running'; });
          const current = active.length ? active[active.length - 1].step_name : 'starting...';
          html += '<div class="step progress">' +
            '<strong style="color:#60a5fa;">&#9679; RUNNING</strong> &nbsp; ' +
            done + '/' + total + ' steps done &nbsp; | &nbsp; current: ' + escapeHtml(current) +
            ' &nbsp; <button class="btn-sm btn-reject" onclick="cancelRun(' + run.id + ')">Cancel</button>' +
            '</div>';
        }

        var lastAttempt = null;
        for (const s of run.steps) {
          // Group trace rows by execution attempt, so a retried run's second pass
          // doesn't interleave with the first (step order restarts per attempt).
          if (run.attempt > 1 && s.run_attempt !== lastAttempt) {
            html += '<div class="section-title">Attempt ' + escapeHtml(s.run_attempt || 1) + '</div>';
            lastAttempt = s.run_attempt;
          }
          const review = s.needs_human_review;
          html += '<div class="step ' + (review ? 'review' : '') + '">' +
            '<strong>' + escapeHtml(s.step_name) + '</strong> [' + escapeHtml(s.status) + ']';
          if (s.security_flag) {
            html += ' <span class="security">&#9888; SECURITY WARNING</span> <span class="reason">[' +
              escapeHtml(s.security_reason || '') + ']</span>';
          }
          if (s.judge_status === 'invalid_output') {
            html += ' <span class="flag">judge returned invalid output</span>';
          }
          if (s.match_score !== null) {
            var finalDec = s.final_decision || s.llm_decision || s.score_decision;
            html += ' - <strong>' + escapeHtml(finalDec) + '</strong>' + ' <span class="call">(score ' + escapeHtml(s.match_score) + ':' + escapeHtml(s.score_decision) + ', judge:' + escapeHtml(s.llm_decision) + ')</span>';
            if (review && s.review_status) {
              html += ' <span class="reviewed">&#9679; ' + escapeHtml(s.review_status.toUpperCase());
              if (s.reviewer) html += ' by ' + escapeHtml(s.reviewer);
              html += '</span>';
              if (s.review_reason) html += ' <span class="reason">[' + escapeHtml(s.review_reason) + ']</span>';
              if (s.review_comment) html += '<div class="call">note: ' + escapeHtml(s.review_comment) + '</div>';
            } else if (review) {
              html += ' <span class="flag">&#9888; NEEDS REVIEW</span>';
              if (s.review_reason) html += ' <span class="reason">[' + escapeHtml(s.review_reason) + ']</span>';
            } else {
              html += ' <span class="agree">&#10003;</span>';
            }
          }
          // Score breakdown — shows WHERE the score came from, key context for review.
          if (s.score_breakdown) {
            html += '<div class="bd">' + renderBreakdown(s.score_breakdown) + '</div>';
          }
          if (s.error_message) {
            html += '<div class="call call-failed">error: ' + escapeHtml(s.error_message) + '</div>';
          }
          for (const t of (s.tool_calls || [])) {
            const fc = t.status === 'failed' ? 'call-failed' : '';
            let io = 'IN: ' + escapeHtml(t.input_json || '') + NL + NL + 'OUT: ' + escapeHtml(t.output_json || '');
            if (t.error_message) io += NL + NL + 'ERROR: ' + escapeHtml(t.error_message);
            html += '<div class="call ' + fc + '"><span class="op">' + escapeHtml(t.operation || t.tool_name) + '</span> &middot; tool: ' + escapeHtml(t.tool_name) + ' [' + escapeHtml(t.status) + '] ' + escapeHtml(t.latency_ms) + 'ms' +
              '<details><summary>view i/o</summary><pre class="io">' + io + '</pre></details></div>';
          }
          for (const l of (s.llm_calls || [])) {
            const fc = l.status === 'failed' ? 'call-failed' : '';
            const attemptLabel = (l.attempt_number && l.attempt_number > 1) ? ' <span class="retry">(attempt ' + escapeHtml(l.attempt_number) + ')</span>' : '';
            let io = 'PROMPT:' + NL + escapeHtml(l.prompt || '') + NL + NL + 'RESPONSE:' + NL + escapeHtml(l.response || '');
            if (l.error_message) io += NL + NL + 'ERROR: ' + escapeHtml(l.error_message);
            if (l.provider_request_id) io += NL + NL + 'REQUEST_ID: ' + escapeHtml(l.provider_request_id);
            html += '<div class="call ' + fc + '"><span class="op">' + escapeHtml(l.operation || 'llm') + '</span> &middot; llm' + attemptLabel + ': ' + escapeHtml(l.prompt_tokens) + '+' + escapeHtml(l.completion_tokens) + ' tok, ' + escapeHtml(l.latency_ms) + 'ms, ~$' + escapeHtml(l.cost_usd) + ' [' + escapeHtml(l.status) + ']' +
              '<details><summary>view prompt/response</summary><pre class="io">' + io + '</pre></details></div>';
          }
          if (s.evaluation) {
            const ev = s.evaluation;
            // An LLM grading an LLM: a quality SIGNAL, not proof of (no) hallucination.
            html += '<div class="call">eval signal (LLM judge): rel=' + escapeHtml(ev.relevance_score) + ' faith=' + escapeHtml(ev.faithfulness_score) + ' complete=' + escapeHtml(ev.completeness_score) + ' halluc=' + escapeHtml(ev.hallucination_detected) + '</div>';
          }
          html += '</div>';
        }
        d.innerHTML = html;

        if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
        if (active) {
          pollTimer = setTimeout(function() { loadDetail(id); }, 3000);
        } else {
          loadRuns();
        }
      } catch (err) {
        d.innerHTML = '<div style="color:#f87171;">Could not load run detail: ' + escapeHtml(err.message) + '</div>';
      }
    }


    // ---- LangGraph human-review flow ----
    async function renderGraphReviews() {
      const box = document.getElementById('graphReviews');
      if (!box) return;
      try {
        const res = await fetch('/runs');
        const runs = await res.json();
        const waiting = runs.filter(function(r){ return r.status === 'waiting_for_human'; });
        if (!waiting.length) { box.innerHTML = ''; return; }
        let html = '<div class="section-title">Runs Awaiting Your Review</div>';
        for (const r of waiting) {
          const detail = await (await fetch('/runs/' + r.id)).json();
          const pr = detail.pending_review || {};
          html += '<div class="step review">' +
            '<strong>Run #' + escapeHtml(r.id) + '</strong> &mdash; ' +
            escapeHtml(pr.job_title || '(job)') +
            ' | score ' + escapeHtml(pr.score) + ' (' + escapeHtml(pr.score_decision) + ')' +
            ' | LLM: ' + escapeHtml(pr.llm_decision) +
            '<div class="reason" style="margin-top:6px;">Why this paused: ' +
              escapeHtml(pr.review_reason || 'flagged for review') + '</div>' +
            '<div style="margin-top:8px;">' +
              '<input id="grc_' + r.id + '" class="rev-input" placeholder="comment (optional)" style="width:260px;">' +
            '</div>' +
            '<div style="margin-top:8px;">' +
              '<button class="btn-sm btn-approve" onclick="resumeRun(' + r.id + ', \'Apply\')">Apply</button>' +
              '<button class="btn-sm" onclick="resumeRun(' + r.id + ', \'Maybe\')">Maybe</button>' +
              '<button class="btn-sm btn-reject" onclick="resumeRun(' + r.id + ', \'Skip\')">Skip</button>' +
            '</div></div>';
        }
        box.innerHTML = html;
      } catch (err) {
        box.innerHTML = '<div class="step" style="color:#f87171;">Could not load reviews: ' + escapeHtml(err.message) + '</div>';
      }
    }

    async function resumeRun(runId, decision) {
      const el = document.getElementById('grc_' + runId);
      const comment = el ? el.value : '';
      const qs = '?decision=' + decision + '&comment=' + encodeURIComponent(comment);
      try {
        const res = await csrfFetch('/runs/' + runId + '/resume' + qs, { method: 'POST' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'resume failed'); }
        setTimeout(function() { loadRuns(); renderGraphReviews(); }, 1200);
      } catch (err) { alert('Error: ' + err.message); }
    }

    // --- Auth gate -----------------------------------------------------------
    var _reviewTimer = null;

    function showLogin(msg) {
      document.getElementById('loginPanel').style.display = 'block';
      document.getElementById('appRoot').style.display = 'none';
      if (msg) document.getElementById('loginMsg').textContent = msg;
    }

    function showApp(user) {
      document.getElementById('loginPanel').style.display = 'none';
      document.getElementById('appRoot').style.display = 'block';
      var who = document.getElementById('whoami');
      if (who) who.textContent = 'Signed in as ' + (user && user.username ? user.username : '');
      // Start the dashboard now that we are authenticated.
      loadResumes();
      renderGraphReviews();
      if (!_reviewTimer) _reviewTimer = setInterval(renderGraphReviews, 4000);
      loadRuns();
    }

    async function doLogin() {
      const u = document.getElementById('loginUser').value.trim();
      const p = document.getElementById('loginPass').value;
      document.getElementById('loginMsg').textContent = '';
      const fd = new FormData();
      fd.append('username', u);
      fd.append('password', p);
      try {
        const res = await csrfFetch('/login', { method: 'POST', body: fd });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'login failed'); }
        const user = await res.json();
        showApp(user);
      } catch (err) {
        document.getElementById('loginMsg').textContent = 'Error: ' + err.message;
      }
    }

    async function doLogout() {
      try { await csrfFetch('/logout', { method: 'POST' }); } catch (e) {}
      _csrfToken = null;   // drop the old session's token; next session fetches a fresh one
      if (_reviewTimer) { clearInterval(_reviewTimer); _reviewTimer = null; }
      showLogin('Signed out.');
    }

    async function initApp() {
      await ensureCsrf();   // get a CSRF token before any state-changing request
      try {
        const res = await fetch('/me');
        if (res.status === 401) { showLogin(); return; }
        if (!res.ok) { showLogin(); return; }
        const user = await res.json();
        showApp(user);
      } catch (err) {
        showLogin();
      }
    }

    initApp();