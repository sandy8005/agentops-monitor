var NL = String.fromCharCode(10);
    var pollTimer = null;

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
        const up = await fetch('/upload', { method: 'POST', body: fd });
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
                '<div><label>Location <span style="color:#8b8f9c;">(informational — does not filter)</span></label><br><input type="text" id="loc_' + r.id + '" placeholder="Michigan" style="width:95%;"></div>' +
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
        const run = await fetch('/runs' + qs, { method: 'POST' });
        if (!run.ok) { const e = await run.json(); throw new Error(e.detail || 'run failed'); }
        const runData = await run.json();
        setTimeout(function() { loadRuns(); loadDetail(runData.run_id); }, 1500);
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function cancelRun(id) {
      if (!confirm('Cancel run #' + id + '?')) return;
      try {
        const res = await fetch('/runs/' + id + '/cancel', { method: 'POST' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'cancel failed'); }
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function deleteResume(id) {
      if (!confirm('Delete resume #' + id + '?')) return;
      try {
        const res = await fetch('/resumes/' + id, { method: 'DELETE' });
        if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'delete failed'); }
        loadResumes();
      } catch (err) { alert('Error: ' + err.message); }
    }

    async function loadPending() {
      const div = document.getElementById('pending');
      try {
        const res = await fetch('/reviews/pending');
        if (!res.ok) throw new Error('failed to load reviews');
        const items = await res.json();
        if (items.length === 0) { div.innerHTML = '<div class="step">No steps pending review.</div>'; return; }
        div.innerHTML = '';
        for (const it of items) {
          const el = document.createElement('div');
          el.className = 'step review';
          el.innerHTML = '<strong>' + escapeHtml(it.step_name) + '</strong> (run #' + escapeHtml(it.run_id) + ') - ' +
            'Score: ' + escapeHtml(it.match_score) + ' (' + escapeHtml(it.score_decision) + ') | LLM: ' + escapeHtml(it.llm_decision) +
            (it.review_reason ? ' <span class="reason">[' + escapeHtml(it.review_reason) + ']</span>' : '') +
            '<div style="margin-top:8px;">' +
              '<input id="reviewer_' + it.step_id + '" class="rev-input" placeholder="your name" style="width:140px;">' +
              '<input id="comment_' + it.step_id + '" class="rev-input" placeholder="comment (optional)" style="width:260px;">' +
            '</div>' +
            '<div style="margin-top:8px;">' +
              '<button class="btn-sm btn-approve" onclick="review(' + it.step_id + ', \'approved\')">Approve</button>' +
              '<button class="btn-sm btn-reject" onclick="review(' + it.step_id + ', \'rejected\')">Reject</button>' +
            '</div>';
          div.appendChild(el);
        }
      } catch (err) {
        div.innerHTML = '<div class="step" style="color:#f87171;">Could not load reviews: ' + escapeHtml(err.message) + '</div>';
      }
    }

    async function review(stepId, decision) {
      const reviewerEl = document.getElementById('reviewer_' + stepId);
      const commentEl = document.getElementById('comment_' + stepId);
      const reviewer = reviewerEl ? reviewerEl.value.trim() : '';
      const comment = commentEl ? commentEl.value.trim() : '';
      const qs = '?decision=' + decision +
                 '&reviewer=' + encodeURIComponent(reviewer || 'anonymous') +
                 '&comment=' + encodeURIComponent(comment);
      try {
        const res = await fetch('/steps/' + stepId + '/review' + qs, { method: 'POST' });
        if (res.ok) { loadPending(); }
        else { const err = await res.json(); alert('Error: ' + (err.detail || 'failed')); }
      } catch (err) { alert('Network error: ' + err.message); }
    }

    async function loadRuns() {
      const tbody = document.querySelector('#runs tbody');
      try {
        const res = await fetch('/runs');
        if (!res.ok) throw new Error('failed to load runs');
        const runs = await res.json();
        tbody.innerHTML = '';
        for (const r of runs) {
          let cls = 'errors';
          if (r.status === 'success') cls = 'success';
          else if (r.status === 'failed') cls = 'failed';
          else if (r.status === 'running') cls = 'running';
          else if (r.status === 'cancelled') cls = 'cancelled';
          else if (r.status === 'no_matches') cls = 'no_matches';
          const searchDesc = (r.target_role || '-') + (r.location ? ' / ' + r.location : '') + (r.work_mode ? ' / ' + r.work_mode : '');
          const tr = document.createElement('tr');
          tr.innerHTML = '<td>#' + escapeHtml(r.id) + '</td>' +
            '<td><span class="status ' + cls + '">' + escapeHtml(r.status) + '</span></td>' +
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
        const running = run.status === 'running';

        let html = '<h2>Run #' + escapeHtml(run.id) + ' - ' + escapeHtml(run.status);
        if (run.target_role) html += ' <span class="note">(' + escapeHtml(run.target_role) + (run.location ? ' / ' + escapeHtml(run.location) : '') + ')</span>';
        html += '</h2>';

        if (running) {
          const active = run.steps.filter(function(s){ return s.status === 'running'; });
          const current = active.length ? active[active.length - 1].step_name : 'starting...';
          html += '<div class="step progress">' +
            '<strong style="color:#60a5fa;">&#9679; RUNNING</strong> &nbsp; ' +
            done + '/' + total + ' steps done &nbsp; | &nbsp; current: ' + escapeHtml(current) +
            ' &nbsp; <button class="btn-sm btn-reject" onclick="cancelRun(' + run.id + ')">Cancel</button>' +
            '</div>';
        }

        for (const s of run.steps) {
          const review = s.needs_human_review;
          html += '<div class="step ' + (review ? 'review' : '') + '">' +
            '<strong>' + escapeHtml(s.step_name) + '</strong> [' + escapeHtml(s.status) + ']';
          if (s.match_score !== null) {
            html += ' - Score: ' + escapeHtml(s.match_score) + ' (' + escapeHtml(s.score_decision) + ') | LLM: ' + escapeHtml(s.llm_decision);
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
            const b = s.score_breakdown;
            html += '<div class="bd">required ' + escapeHtml(b.required) + '/50 &nbsp; ' +
                    'preferred ' + escapeHtml(b.preferred) + '/20 &nbsp; ' +
                    'projects ' + escapeHtml(b.projects) + '/15 &nbsp; ' +
                    'experience ' + escapeHtml(b.experience) + '/15</div>';
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
            html += '<div class="call">eval: rel=' + escapeHtml(ev.relevance_score) + ' faith=' + escapeHtml(ev.faithfulness_score) + ' complete=' + escapeHtml(ev.completeness_score) + ' halluc=' + escapeHtml(ev.hallucination_detected) + '</div>';
          }
          html += '</div>';
        }
        d.innerHTML = html;

        if (pollTimer) { clearTimeout(pollTimer); pollTimer = null; }
        if (running) {
          pollTimer = setTimeout(function() { loadDetail(id); }, 3000);
        } else {
          loadRuns();
        }
      } catch (err) {
        d.innerHTML = '<div style="color:#f87171;">Could not load run detail: ' + escapeHtml(err.message) + '</div>';
      }
    }

    loadResumes();
    loadPending();
    loadRuns();