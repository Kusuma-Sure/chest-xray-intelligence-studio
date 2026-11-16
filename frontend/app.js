/**
 * Chest X-ray Intelligence Studio — Frontend Controller
 * Manages 5 tabs, 1-5 multi-image uploads, OOD rejection, model status,
 * 4-panel XAI dashboard, and GraphRAG evidence engine.
 */

const $ = (id) => document.getElementById(id);

// Application state
let uploadedFiles = [];
let activeFileIndex = 0;
let registeredModels = [];
let activePrediction = null;

// ==============================================================================
// 1. TOAST NOTIFICATIONS & UTILITIES
// ==============================================================================

function showToast(message, duration = 4000) {
  const container = $('toast-container');
  const toast = document.createElement('div');
  toast.className = 'studio-toast';
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

// ==============================================================================
// 2. TAB SWITCHING (5 TABS)
// ==============================================================================

function switchTab(targetTab) {
  document.querySelectorAll('.nav-tab').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.tab === targetTab);
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('active', panel.id === `panel-${targetTab}`);
  });

  if (targetTab === 'benchmark') {
    loadBenchmarks();
  }
}

document.querySelectorAll('.nav-tab').forEach((tab) => {
  tab.addEventListener('click', () => switchTab(tab.dataset.tab));
});

$('jump-to-xai-btn')?.addEventListener('click', () => switchTab('xai'));

// ==============================================================================
// 3. MULTI-IMAGE BATCH MANAGEMENT (1 to 5 IMAGES)
// ==============================================================================

function handleFilesAdded(fileList) {
  if (!fileList || !fileList.length) return;

  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const newFiles = Array.from(fileList).filter((f) => {
    if (!validTypes.includes(f.type)) {
      showToast(`Skipped ${f.name}: only JPG, PNG, and WEBP are supported.`);
      return false;
    }
    if (f.size > 25 * 1024 * 1024) {
      showToast(`Skipped ${f.name}: exceeds 25 MB size limit.`);
      return false;
    }
    return true;
  });

  if (!newFiles.length) return;

  // Append up to a maximum of 5 images
  for (const f of newFiles) {
    if (uploadedFiles.length < 5) {
      uploadedFiles.push(f);
    } else {
      showToast('Maximum of 5 simultaneous images reached.');
      break;
    }
  }

  activeFileIndex = Math.max(0, uploadedFiles.length - 1);
  renderBatchThumbnails();
  updateInferenceButtonState();
}

function removeFileAtIndex(index) {
  uploadedFiles.splice(index, 1);
  if (activeFileIndex >= uploadedFiles.length) {
    activeFileIndex = Math.max(0, uploadedFiles.length - 1);
  }
  renderBatchThumbnails();
  updateInferenceButtonState();
  if (uploadedFiles.length === 0) {
    resetPredictionView();
  }
}

function renderBatchThumbnails() {
  const container = $('batch-preview-container');
  const statusText = $('upload-status-text');
  const clearBtn = $('clear-all-images');

  if (uploadedFiles.length === 0) {
    container.hidden = true;
    container.innerHTML = '';
    statusText.textContent = 'No image selected';
    clearBtn.disabled = true;
    return;
  }

  container.hidden = false;
  container.innerHTML = '';
  clearBtn.disabled = false;
  statusText.textContent = `${uploadedFiles.length} image(s) selected (inspecting #${activeFileIndex + 1})`;

  uploadedFiles.forEach((file, idx) => {
    const box = document.createElement('div');
    box.className = `thumbnail-box ${idx === activeFileIndex ? 'active' : ''}`;
    box.title = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;

    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    box.appendChild(img);

    const removeBtn = document.createElement('button');
    removeBtn.className = 'thumb-remove-btn';
    removeBtn.innerHTML = '×';
    removeBtn.title = 'Remove image';
    removeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      removeFileAtIndex(idx);
    });
    box.appendChild(removeBtn);

    box.addEventListener('click', () => {
      activeFileIndex = idx;
      renderBatchThumbnails();
    });

    container.appendChild(box);
  });
}

$('image-files').addEventListener('change', (e) => handleFilesAdded(e.target.files));

$('clear-all-images').addEventListener('click', () => {
  uploadedFiles = [];
  activeFileIndex = 0;
  $('image-files').value = '';
  renderBatchThumbnails();
  resetPredictionView();
  updateInferenceButtonState();
});

// Drag and drop handlers
const dropzone = $('dropzone');
['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', (e) => {
  handleFilesAdded(e.dataTransfer.files);
});

// Clipboard paste handler: allows user to press Ctrl+V to paste radiographs directly
window.addEventListener('paste', (e) => {
  const clipboardData = e.clipboardData || window.clipboardData;
  if (!clipboardData) return;
  const items = clipboardData.items;
  if (!items) return;

  const files = [];
  for (let i = 0; i < items.length; i++) {
    if (items[i].type && items[i].type.indexOf('image') !== -1) {
      const file = items[i].getAsFile();
      if (file) {
        const ext = file.type === 'image/jpeg' ? 'jpg' : (file.type === 'image/webp' ? 'webp' : 'png');
        const namedFile = new File([file], `pasted_radiograph_${Date.now()}.${ext}`, { type: file.type || 'image/png' });
        files.push(namedFile);
      }
    }
  }

  if (files.length > 0) {
    e.preventDefault();
    handleFilesAdded(files);
    showToast(`📋 Pasted ${files.length} radiograph(s) from clipboard! Ready to analyze.`);
    switchTab('analysis');
  }
});

// ==============================================================================
// 4. MODEL SELECTION & STATUS (CRITICAL BUG FIX)
// ==============================================================================

async function loadSystemStatus() {
  try {
    const [healthRes, modelsRes] = await Promise.all([
      fetch('/api/health'),
      fetch('/api/models')
    ]);

    if (!healthRes.ok || !modelsRes.ok) throw new Error('API request failed');

    const health = await healthRes.json();
    const modelsData = await modelsRes.json();
    registeredModels = modelsData.models;

    $('runtime-device').textContent = `${health.device} (${health.models_with_checkpoints}/${health.models_total} Checkpoints)`;
    $('api-status').textContent = 'API Online';
    $('api-status').className = 'status-online';

    // Populate model select dropdown without disabling options
    const select = $('model-select');
    select.innerHTML = registeredModels.map((m) => {
      const statusTag = m.available ? '✓ AVAILABLE' : '⚠️ MISSING CHECKPOINT';
      const primaryTag = m.primary ? ' [PRIMARY]' : '';
      return `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)}${primaryTag} · ${statusTag}</option>`;
    }).join('');

    // Default to primary model ViT or first available
    const primary = registeredModels.find((m) => m.primary) || registeredModels[0];
    if (primary) {
      select.value = primary.id;
    }

    onModelChanged();
  } catch (err) {
    $('api-status').textContent = 'API Offline';
    $('api-status').className = 'status-offline';
    $('runtime-device').textContent = 'Disconnected';
    $('model-select').innerHTML = '<option value="">Backend API unavailable</option>';
    $('model-help-text').textContent = 'Please run app.py on Windows to connect to the backend.';
    updateInferenceButtonState();
  }
}

function onModelChanged() {
  const modelId = $('model-select').value;
  const model = registeredModels.find((m) => m.id === modelId);
  const badge = $('model-status-badge');
  const helpText = $('model-help-text');
  const noticeBox = $('checkpoint-missing-notice');

  if (!model) return;

  if (model.available) {
    badge.textContent = 'CHECKPOINT LOADED';
    badge.className = 'model-badge badge-ready';
    helpText.textContent = `${model.family} · ${model.explanation_method}`;
    noticeBox.hidden = true;
  } else {
    badge.textContent = 'CHECKPOINT MISSING';
    badge.className = 'model-badge badge-missing';
    helpText.textContent = model.reason;
    $('missing-notice-title').textContent = `${model.name} — Checkpoint Missing`;
    $('missing-notice-text').textContent = model.reason;
    noticeBox.hidden = false;
  }

  updateInferenceButtonState();
}

$('model-select').addEventListener('change', onModelChanged);

function updateInferenceButtonState() {
  const modelId = $('model-select').value;
  const model = registeredModels.find((m) => m.id === modelId);
  const btn = $('run-inference-btn');
  const hint = $('action-hint');

  const hasImages = uploadedFiles.length > 0;
  const isReady = !!model?.available;

  btn.disabled = !hasImages || !isReady;

  if (!hasImages) {
    hint.textContent = 'Upload 1 to 5 frontal chest radiographs to begin.';
  } else if (!isReady) {
    hint.textContent = `Selected architecture '${model?.name || 'model'}' requires training in Google Colab before inference.`;
  } else {
    hint.textContent = `Ready to run inference using ${model.name} across ${uploadedFiles.length} radiograph(s).`;
  }
}

// Decision Threshold Slider
$('threshold-slider').addEventListener('input', (e) => {
  const val = parseFloat(e.target.value).toFixed(2);
  $('threshold-display').textContent = val;
  if (activePrediction && activePrediction.findings) {
    reRenderFindingsWithThreshold(parseFloat(val));
  }
});

// ==============================================================================
// 5. INFERENCE & PREDICTION CONTROLLER
// ==============================================================================

$('run-inference-btn').addEventListener('click', async () => {
  if (!uploadedFiles.length) return;

  const modelId = $('model-select').value;
  const threshold = $('threshold-slider').value;
  const btn = $('run-inference-btn');

  btn.disabled = true;
  btn.innerHTML = '<span class="btn-text">Analyzing Radiograph…</span><span class="spinner">◌</span>';

  // Reset alert states
  $('ood-alert-banner').hidden = true;
  $('checkpoint-missing-notice').hidden = true;

  try {
    const formData = new FormData();
    uploadedFiles.forEach((file) => {
      formData.append('image', file);
    });

    const url = `/api/predict?model=${encodeURIComponent(modelId)}&threshold=${threshold}&explain=true`;
    const res = await fetch(url, { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || 'Inference request failed');
    }

    // Check for Out-of-Distribution rejection
    if (data.is_ood) {
      handleOODRejection(data);
      return;
    }

    // Check for missing model weights
    if (data.model_unavailable) {
      showToast(data.user_message, 6000);
      $('checkpoint-missing-notice').hidden = false;
      return;
    }

    activePrediction = data;
    renderPredictionResults(data);
    showToast('Inference and explainability pipeline completed successfully.');
  } catch (err) {
    showToast(`Error: ${err.message}`, 5000);
  } finally {
    btn.innerHTML = '<span class="btn-text">Analyze Chest Radiograph</span><span class="btn-arrow">→</span>';
    updateInferenceButtonState();
  }
});

function handleOODRejection(data) {
  $('empty-results-state').hidden = true;
  $('results-display-container').hidden = true;
  const banner = $('ood-alert-banner');
  banner.hidden = false;

  $('ood-alert-title').textContent = 'Input Rejected: Non-Medical or Unsupported Image Detected';
  $('ood-alert-message').textContent = data.user_message;

  const diag = data.ood_details?.diagnostics || {};
  $('ood-diagnostics-box').innerHTML = `
    <strong>Diagnostics:</strong>
    OOD Score: ${data.ood_details?.ood_score} (Cutoff: ${data.ood_details?.threshold}) |
    Color Saturation: ${diag.mean_saturation ?? 'N/A'} |
    Entropy: ${diag.entropy ?? 'N/A'} |
    Reason: ${escapeHtml(data.rejection_reason || 'Out-of-distribution')}
  `;

  showToast('Image rejected: Not a valid frontal chest radiograph.', 5000);
}

function resetPredictionView() {
  activePrediction = null;
  $('empty-results-state').hidden = false;
  $('results-display-container').hidden = true;
  $('ood-alert-banner').hidden = true;
  $('xai-empty-notice').hidden = false;
  $('xai-active-grid').hidden = true;
  $('xai-methodology-note').hidden = true;
}

function renderPredictionResults(data) {
  $('empty-results-state').hidden = true;
  $('ood-alert-banner').hidden = true;
  $('results-display-container').hidden = false;

  $('result-model-name').textContent = data.model_name;
  $('result-runtime').textContent = `${data.processing_time_ms.toFixed(1)} ms`;
  $('result-threshold-val').textContent = parseFloat(data.threshold).toFixed(2);

  const threshold = parseFloat($('threshold-slider').value);
  reRenderFindingsWithThreshold(threshold);

  // Populate 4-Panel XAI
  if (data.explanation && data.explanation.available && data.explanation.panels) {
    renderFourPanelXAI(data.explanation.panels, data.explanation.method);
  }

  // Populate GraphRAG Evidence
  if (data.graph_rag) {
    renderGraphRAGEvidence(data.graph_rag);
  }
}

function reRenderFindingsWithThreshold(threshold) {
  if (!activePrediction || !activePrediction.findings) return;

  const findings = activePrediction.findings.map((f) => ({
    ...f,
    positive: f.probability >= threshold,
  }));

  const positiveCount = findings.filter((f) => f.positive).length;
  $('result-positives-count').textContent = `${positiveCount} Detected`;
  $('result-threshold-val').textContent = threshold.toFixed(2);

  const list = $('findings-list');
  list.innerHTML = findings
    .sort((a, b) => b.probability - a.probability)
    .map((f) => {
      const pct = (f.probability * 100).toFixed(1);
      return `
        <div class="finding-row ${f.positive ? 'positive-row' : ''}">
          <span class="finding-name">${escapeHtml(f.finding)}</span>
          <div class="finding-progress-track">
            <div class="finding-progress-bar" style="width: ${Math.min(100, Math.max(0, f.probability * 100))}%"></div>
          </div>
          <span class="finding-prob-text">${pct}%</span>
        </div>
      `;
    })
    .join('');

  // Update XAI focus dropdown with findings
  const xaiSelect = $('xai-focus-select');
  xaiSelect.innerHTML = findings.map((f, idx) => `
    <option value="${f.index}" ${f.finding === activePrediction.highest_finding ? 'selected' : ''}>
      ${escapeHtml(f.finding)} (${(f.probability * 100).toFixed(1)}%)
    </option>
  `).join('');
}

// ==============================================================================
// 6. 4-PANEL XAI DASHBOARD
// ==============================================================================

function renderFourPanelXAI(panels, method) {
  $('xai-empty-notice').hidden = true;
  $('xai-active-grid').hidden = false;
  $('xai-methodology-note').hidden = false;

  $('panel-1-img').src = panels.panel_1_original;
  $('panel-2-img').src = panels.panel_2_heatmap;
  $('panel-2-title').textContent = method || 'Model-Specific Heatmap';

  $('panel-3-img').src = panels.panel_3_rollout;
  $('panel-3-title').textContent = panels.panel_3_title || 'ViT Attention Rollout';

  $('panel-4-img').src = panels.panel_4_fused;
  $('panel-4-title').textContent = `Fused Overlay: ${panels.finding} (${(panels.probability * 100).toFixed(1)}%)`;
}

$('xai-focus-select')?.addEventListener('change', async (e) => {
  const findingIdx = parseInt(e.target.value, 10);
  if (isNaN(findingIdx) || !uploadedFiles.length) return;

  const modelId = $('model-select').value;
  const threshold = $('threshold-slider').value;

  try {
    const formData = new FormData();
    formData.append('image', uploadedFiles[activeFileIndex]);

    const url = `/api/predict?model=${encodeURIComponent(modelId)}&threshold=${threshold}&explain=true&target_finding_idx=${findingIdx}`;
    const res = await fetch(url, { method: 'POST', body: formData });
    const data = await res.json();

    if (data.explanation?.available && data.explanation.panels) {
      renderFourPanelXAI(data.explanation.panels, data.explanation.method);
      if (data.graph_rag) {
        renderGraphRAGEvidence(data.graph_rag);
      }
      showToast(`Attribution updated for ${data.explanation.target_finding}`);
    }
  } catch (err) {
    showToast(`Failed to update attribution: ${err.message}`);
  }
});

// ==============================================================================
// 7. GRAPHRAG EVIDENCE & CITATIONS
// ==============================================================================

function renderGraphRAGEvidence(graphData) {
  const explanation = graphData.grounded_explanation;
  if (!explanation) return;

  // Render markdown report
  const reportBox = $('grounded-synthesis-text');
  reportBox.innerHTML = `
    <div class="synthesis-block">
      <p><strong>Status:</strong> ${escapeHtml(explanation.summary)}</p>
      <div class="grounded-prose-text">${escapeHtml(explanation.grounded_text).replace(/\n\n/g, '<br><br>')}</div>
    </div>
  `;

  // Render citations
  const citationsList = $('evidence-citations-list');
  const citations = explanation.retrieved_evidence || [];
  if (citations.length) {
    citationsList.innerHTML = citations.map((c) => `
      <div class="citation-card">
        <div class="citation-title">
          <a href="${escapeHtml(c.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(c.title)}</a>
          <small>· ${escapeHtml(c.source)}</small>
        </div>
        <p class="citation-excerpt">"${escapeHtml(c.excerpt)}"</p>
      </div>
    `).join('');
  } else {
    citationsList.innerHTML = '<p class="subtle-text">The supplied evidence does not establish external literature links for this finding.</p>';
  }

  // Render anatomical tags
  const tagsBox = $('anatomical-regions-box');
  const regions = explanation.anatomical_regions || [];
  tagsBox.innerHTML = regions.map((r) => `<span class="tag-pill">🫁 ${escapeHtml(r)}</span>`).join('') || '<span class="tag-pill">Thoracic Cavity</span>';

  // Render relations viewport
  const relationsViewport = $('graph-relations-viewport');
  const relations = graphData.relations || [];
  if (relations.length) {
    relationsViewport.innerHTML = relations.slice(0, 20).map((r) => `
      <div class="relation-item">
        <span class="rel-node">${escapeHtml(r.source)}</span>
        <span class="rel-edge">──[ ${escapeHtml(r.relation)} ]──▶</span>
        <span class="rel-node">${escapeHtml(r.target)}</span>
      </div>
    `).join('');
  }
}

// Browse specific finding evidence
async function loadFindingEvidence(findingName) {
  try {
    const res = await fetch(`/api/graph-evidence?finding=${encodeURIComponent(findingName)}`);
    if (!res.ok) throw new Error('Evidence fetch failed');
    const data = await res.json();
    renderGraphRAGEvidence(data);
  } catch (err) {
    showToast(`Could not load evidence for ${findingName}`);
  }
}

// Populate Finding Filter Dropdown in GraphRAG Tab
const standardFindings = [
  "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
  "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
  "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
];

const findingSelect = $('graph-finding-filter');
if (findingSelect) {
  findingSelect.innerHTML = standardFindings.map((f) => `<option value="${f}">${f}</option>`).join('');
  findingSelect.addEventListener('change', (e) => loadFindingEvidence(e.target.value));
}

// ==============================================================================
// 8. BENCHMARKS & EVALUATION REGISTER
// ==============================================================================

async function loadBenchmarks() {
  const tbody = $('benchmark-table-body');
  tbody.innerHTML = '<tr><td colspan="9" class="table-empty">Loading benchmark metrics…</td></tr>';

  try {
    const res = await fetch('/api/benchmarks');
    if (!res.ok) throw new Error('Could not fetch benchmarks');
    const data = await res.json();

    tbody.innerHTML = data.models.map((m) => {
      const fmt = (v) => (v !== null && v !== undefined && !isNaN(v)) ? Number(v).toFixed(4) : '—';
      const isPrimary = m.primary ? ' <span class="badge-primary">[PRIMARY]</span>' : '';
      return `
        <tr>
          <td><strong>${escapeHtml(m.name)}</strong>${isPrimary}</td>
          <td>${escapeHtml(m.family)}</td>
          <td><strong>${fmt(m.macro_auroc)}</strong></td>
          <td>${fmt(m.macro_auprc)}</td>
          <td>${fmt(m.micro_f1)}</td>
          <td>${fmt(m.macro_f1)}</td>
          <td>${fmt(m.micro_accuracy)}</td>
          <td>${m.validation_samples ?? '200'}</td>
          <td><span class="badge-ready">${escapeHtml(m.status)}</span></td>
        </tr>
      `;
    }).join('');

    $('benchmark-provenance-note').textContent = data.provenance || 'Empirically evaluated on CheXpert.';

    if (data.comparison_chart) {
      $('benchmark-chart-overall').src = data.comparison_chart;
    }
    if (data.finding_chart) {
      $('benchmark-chart-findings').src = data.finding_chart;
    }
  } catch (err) {
    tbody.innerHTML = '<tr><td colspan="9" class="table-empty">Benchmark API unavailable.</td></tr>';
  }
}

$('refresh-benchmarks-btn')?.addEventListener('click', loadBenchmarks);

// ==============================================================================
// INITIALIZATION
// ==============================================================================

window.addEventListener('DOMContentLoaded', () => {
  loadSystemStatus();
  loadBenchmarks();
  loadFindingEvidence("Lung Opacity");
});
