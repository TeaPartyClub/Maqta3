  /* ── State ─────────────────────────────── */
  let activeTab = 'url';
  let uploadedFile = null;
  let jobId = null;
  let pollTimer = null;
  let selectedClips = 5;

  function selectClips(n) {
    selectedClips = n;
    document.querySelectorAll('.clip-btn').forEach((btn, i) => {
      btn.classList.toggle('selected', i + 1 === n);
    });
  }

  /* ── Pipeline stages (ordered) ─────────── */
  const STAGES = [
    { id: 'downloading',  label: 'Downloading video' },
    { id: 'extracting',   label: 'Extracting audio track' },
    { id: 'transcribing', label: 'Transcribing speech  (Whisper AI)' },
    { id: 'scoring',      label: 'Scoring highlight candidates' },
    { id: 'clipping',     label: 'Cutting clips' },
    { id: 'converting',   label: 'Converting to vertical 9:16' },
    { id: 'subtitling',   label: 'Burning Arabic subtitles' },
    { id: 'dialect',      label: 'Rewriting to Saudi dialect' },
    { id: 'tts',          label: 'Generating Arabic voiceover' },
    { id: 'done',         label: 'Finalising output' },
  ];

  /* ── Tab switching ──────────────────────── */
  function switchTab(tab) {
    activeTab = tab;
    document.getElementById('tab-url').classList.toggle('active', tab === 'url');
    document.getElementById('tab-upload').classList.toggle('active', tab === 'upload');
    document.getElementById('panel-url').classList.toggle('hidden', tab !== 'url');
    document.getElementById('panel-upload').classList.toggle('hidden', tab !== 'upload');
  }

  /* ── File handling ──────────────────────── */
  function setFile(file) {
    uploadedFile = file;
    const el = document.getElementById('file-name');
    el.style.display = 'block';
    el.textContent = `✓  ${file.name}  (${(file.size / 1048576).toFixed(1)} MB)`;
  }

  function onFileSelect(e) { if (e.target.files[0]) setFile(e.target.files[0]); }

  function onDragOver(e) {
    e.preventDefault();
    document.getElementById('drop-zone').classList.add('active');
  }
  function onDragLeave(e) {
    document.getElementById('drop-zone').classList.remove('active');
  }
  function onDrop(e) {
    e.preventDefault();
    document.getElementById('drop-zone').classList.remove('active');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setFile(f);
  }

  /* ── Submit ─────────────────────────────── */
  async function submitJob() {
    const fd = new FormData();
    fd.append('num_highlights', selectedClips);

    if (activeTab === 'url') {
      const url = document.getElementById('yt-url').value.trim();
      if (!url) { alert('Please enter a YouTube URL.'); return; }
      if (!url.includes('youtube.com') && !url.includes('youtu.be')) {
        alert('Only YouTube URLs are supported.'); return;
      }
      fd.append('url', url);
    } else {
      if (!uploadedFile) { alert('Please select a video file.'); return; }
      fd.append('file', uploadedFile);
    }

    showStatus();
    renderStages(null);

    try {
      const res = await fetch('/api/jobs', { method: 'POST', body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      jobId = data.job_id;
      pollTimer = setInterval(poll, 2500);
    } catch (err) {
      showError(err.message);
    }
  }

  /* ── Poll ───────────────────────────────── */
  async function poll() {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const data = await res.json();
      updateProgress(data);
      if (data.status === 'done')  { clearInterval(pollTimer); showResults(data.results); }
      if (data.status === 'error') { clearInterval(pollTimer); showError(data.error || 'Unknown error'); }
    } catch (e) {
      /* keep polling on transient network errors */
    }
  }

  /* ── UI helpers ─────────────────────────── */
  function showStatus() {
    document.getElementById('input-section').style.display = 'none';
    document.getElementById('status-section').style.display = 'block';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('err-box').style.display = 'none';
  }

  function updateProgress(data) {
    document.getElementById('prog-bar').style.width = (data.progress || 0) + '%';
    document.getElementById('status-title').textContent = data.message || 'Processing…';
    renderStages(data.stage);
  }

  function renderStages(active) {
    const idx = STAGES.findIndex(s => s.id === active);
    document.getElementById('stage-list').innerHTML = STAGES.map((s, i) => {
      let cls = '', dot = '○';
      if (idx >= 0) {
        if (i < idx)  { cls = 's-done';   dot = '✓'; }
        if (i === idx){ cls = 's-active'; dot = '●'; }
      }
      return `<div class="stage-row ${cls}">
        <div class="stage-dot">${dot}</div>
        <div class="stage-label">${s.label}</div>
      </div>`;
    }).join('');
  }

  function showError(msg) {
    const el = document.getElementById('err-box');
    el.textContent = '⚠  ' + msg;
    el.style.display = 'block';
    document.getElementById('status-title').textContent = 'Processing failed';
    setTimeout(() => {
      clearInterval(pollTimer);
      document.getElementById('input-section').style.display = 'block';
      document.getElementById('status-section').style.display = 'none';
    }, 4000);
  }

  function showResults(clips) {
    document.getElementById('status-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'block';
    document.getElementById('clip-count').textContent = clips.length;

    document.getElementById('results-grid').innerHTML = clips.map((c, i) => `
      <div class="clip-card">
        <video class="clip-video" controls playsinline preload="metadata">
          <source src="/api/results/${jobId}/${c.filename}" type="video/mp4">
        </video>
        <div class="clip-body">
          <span class="clip-badge">Clip ${i + 1} · ${fmtDur(c.duration)}</span>
          <p class="clip-preview">${c.arabic_preview || ''}</p>
          <a class="dl-btn" href="/api/results/${jobId}/${c.filename}" download>
            ⬇ &nbsp; Download Clip ${i + 1}
          </a>
        </div>
      </div>
    `).join('');
  }

  function fmtDur(sec) {
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m > 0 ? `${m}m ${s}s` : `${s}s`;
  }

  function resetUI() {
    clearInterval(pollTimer);
    jobId = null; uploadedFile = null;
    document.getElementById('yt-url').value = '';
    document.getElementById('file-name').style.display = 'none';
    document.getElementById('file-input').value = '';
    document.getElementById('results-grid').innerHTML = '';
    document.getElementById('prog-bar').style.width = '0%';
    document.getElementById('err-box').style.display = 'none';
    document.getElementById('input-section').style.display = 'block';
    document.getElementById('status-section').style.display = 'none';
    document.getElementById('results-section').style.display = 'none';
  }
