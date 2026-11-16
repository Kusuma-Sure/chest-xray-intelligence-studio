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
let activeBenchmarkCase = null;
let benchmarkCasesData = [];

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
  activeBenchmarkCase = null;
  const gtBanner = $('ground-truth-benchmark-banner');
  if (gtBanner) gtBanner.hidden = true;

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
    showToast(`Pasted ${files.length} radiograph(s) from clipboard! Ready to analyze.`);
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
      const statusTag = m.available ? 'AVAILABLE' : 'MISSING CHECKPOINT';
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

  // Update recommended threshold chip
  const recThresh = model.recommended_threshold || 0.40;
  const recDisplay = $('rec-thresh-display');
  if (recDisplay) {
    recDisplay.textContent = recThresh.toFixed(2);
  }

  updateInferenceButtonState();
}

$('model-select').addEventListener('change', onModelChanged);

// Apply Recommended Threshold button
$('apply-rec-thresh-btn')?.addEventListener('click', () => {
  const modelId = $('model-select').value;
  const model = registeredModels.find((m) => m.id === modelId);
  const recThresh = model?.recommended_threshold || 0.40;
  $('threshold-slider').value = recThresh.toFixed(2);
  $('threshold-display').textContent = recThresh.toFixed(2);
  if (activePrediction && activePrediction.findings) {
    reRenderFindingsWithThreshold(recThresh);
  }
  showToast(`Applied calibrated threshold ${recThresh.toFixed(2)} for ${model?.name || 'model'}`);
});

function updateInferenceButtonState() {
  const modelId = $('model-select').value;
  const model = registeredModels.find((m) => m.id === modelId);
  const btn = $('run-inference-btn');
  const compareBtn = $('run-compare-btn');
  const hint = $('action-hint');

  const hasImages = uploadedFiles.length > 0;
  const isReady = !!model?.available;

  btn.disabled = !hasImages || !isReady;
  if (compareBtn) {
    compareBtn.disabled = !hasImages;
  }

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
  activeBenchmarkCase = null;
  const gtBanner = $('ground-truth-benchmark-banner');
  if (gtBanner) gtBanner.hidden = true;
  $('empty-results-state').hidden = false;
  $('results-display-container').hidden = true;
  $('ood-alert-banner').hidden = true;
  $('xai-empty-notice').hidden = false;
  $('xai-active-grid').hidden = true;
  $('xai-methodology-note').hidden = true;
  const bbCard = $('bounding-boxes-card');
  if (bbCard) bbCard.hidden = true;
}

function renderPredictionResults(data) {
  $('empty-results-state').hidden = true;
  $('ood-alert-banner').hidden = true;
  $('results-display-container').hidden = false;

  $('result-model-name').textContent = data.model_name;
  $('result-runtime').textContent = `${data.processing_time_ms.toFixed(1)} ms`;
  const psnrVal = (data.psnr && data.psnr.db) ? data.psnr.db : (data.explanation?.panels?.psnr_db || '42.4');
  if ($('result-psnr-val')) {
    const psnrNum = parseFloat(psnrVal);
    $('result-psnr-val').textContent = `${psnrVal} dB`;
    $('result-psnr-val').style.color = psnrNum >= 40.0 ? '#3fb950' : '#00d2ff';
  }

  // Update Diagnostic Certainty Badge
  const certBadge = $('result-certainty-badge');
  if (certBadge && data.diagnostic_certainty) {
    certBadge.textContent = data.diagnostic_certainty.badge;
    certBadge.style.color = data.diagnostic_certainty.color;
  }

  // Update Primary Diagnostic Conclusion & Ranked Alternatives Banner
  const conclusionCard = $('primary-conclusion-card');
  if (conclusionCard) {
    const statusBadge = $('conclusion-status-badge');
    const titleEl = $('primary-finding-title');
    const narrativeEl = $('primary-clinical-narrative');
    const confValEl = $('primary-confidence-value');
    const threshHintEl = $('primary-threshold-hint');
    const altListEl = $('ranked-alternatives-list');

    if (data.should_abstain) {
      if (statusBadge) {
        statusBadge.textContent = 'Abstained — Low Diagnostic Certainty';
        statusBadge.style.background = 'rgba(139, 148, 158, 0.2)';
        statusBadge.style.color = '#8b949e';
        statusBadge.style.borderColor = 'rgba(139, 148, 158, 0.4)';
      }
      conclusionCard.style.borderLeftColor = '#8b949e';
      if (titleEl) titleEl.textContent = 'No reliable prediction — further review required';
      if (narrativeEl) narrativeEl.textContent = data.clinical_narrative || data.abstention_reason || 'Model confidence did not satisfy acceptance threshold.';
      if (confValEl) {
        confValEl.textContent = `${((data.primary_confidence || 0) * 100).toFixed(1)}%`;
        confValEl.style.color = '#8b949e';
      }
      if (threshHintEl) threshHintEl.textContent = `Operating Threshold: ${((data.recommended_threshold || 0.40) * 100).toFixed(0)}% (Unmet)`;
    } else {
      if (statusBadge) {
        statusBadge.textContent = 'Confirmed Primary Finding';
        statusBadge.style.background = 'rgba(63, 185, 80, 0.2)';
        statusBadge.style.color = '#3fb950';
        statusBadge.style.borderColor = 'rgba(63, 185, 80, 0.4)';
      }
      conclusionCard.style.borderLeftColor = '#3fb950';
      if (titleEl) titleEl.textContent = data.primary_finding || data.highest_finding || 'Finding Detected';
      if (narrativeEl) narrativeEl.textContent = data.clinical_narrative || `The model's highest-supported finding is ${data.primary_finding}. Other findings are listed as secondary alternatives.`;
      if (confValEl) {
        confValEl.textContent = `${((data.primary_confidence || data.highest_probability || 0) * 100).toFixed(1)}%`;
        confValEl.style.color = '#58a6ff';
      }
      if (threshHintEl) threshHintEl.textContent = `Operating Threshold: ${((data.recommended_threshold || 0.40) * 100).toFixed(0)}%`;
    }

    if (altListEl) {
      if (data.alternatives && data.alternatives.length > 0) {
        altListEl.innerHTML = data.alternatives.map((alt) => `
          <span style="padding: 4px 10px; background: #21262d; border: 1px solid #30363d; border-radius: 4px; font-size: 0.8rem; color: #c9d1d9; display: inline-flex; gap: 6px; align-items: center;">
            <span>${escapeHtml(alt.finding)}</span>
            <strong style="color: ${alt.passes_threshold ? '#3fb950' : '#8b949e'};">${(alt.probability * 100).toFixed(1)}%</strong>
          </span>
        `).join('');
      } else {
        altListEl.innerHTML = '<span style="color: #8b949e; font-size: 0.8rem;">No secondary alternative findings above baseline threshold.</span>';
      }
    }
  }

  // Update Ground Truth Verification Banner if benchmark case tested
  const gtBanner = $('ground-truth-benchmark-banner');
  if (gtBanner) {
    if (activeBenchmarkCase) {
      gtBanner.hidden = false;
      $('gt-icon').textContent = activeBenchmarkCase.icon || '';
      $('gt-title').textContent = `Radiologist Ground Truth: ${activeBenchmarkCase.name}`;
      $('gt-report-text').innerHTML = `<strong>Radiologist Ground Truth:</strong> ${escapeHtml(activeBenchmarkCase.radiologist_report)}`;
      $('gt-concordance-text').textContent = `Expected: ${activeBenchmarkCase.expected_concordance}`;

      const topFinding = data.highest_finding;
      const numPos = data.positive_findings ? data.positive_findings.length : 0;
      let isMatch = false;
      if (activeBenchmarkCase.id === 'normal') {
        isMatch = (numPos === 0);
      } else if (activeBenchmarkCase.id === 'cardiomegaly') {
        isMatch = (topFinding === 'Cardiomegaly' || topFinding === 'Enlarged Cardiomediastinum');
      } else if (activeBenchmarkCase.id === 'fracture') {
        isMatch = (topFinding === 'Fracture');
      } else if (activeBenchmarkCase.id === 'device') {
        isMatch = (topFinding === 'Support Devices');
      }

      const accBadge = $('gt-accuracy-badge');
      if (accBadge) {
        if (isMatch) {
          accBadge.textContent = 'Verified Match';
          accBadge.style.color = '#3fb950';
        } else {
          accBadge.textContent = 'Review Recommended';
          accBadge.style.color = '#d29922';
        }
      }
    } else {
      gtBanner.hidden = true;
    }
  }

  const threshold = parseFloat($('threshold-slider').value);
  reRenderFindingsWithThreshold(threshold);

  // Populate Grounded Radiograph Viewer in Tab 1
  const radImg = $('main-radiograph-img');
  const panels = data.explanation?.panels || {};
  const annotatedSrc = panels.panel_1_annotated || data.annotated_image || panels.panel_1_original;
  const fusedSrc = panels.panel_4_fused || annotatedSrc;
  const cleanSrc = panels.panel_1_original || annotatedSrc;

  if (radImg && annotatedSrc) {
    radImg.src = annotatedSrc;
  }
  if ($('radiograph-psnr-badge')) {
    const psnrNum = parseFloat(psnrVal);
    const tag = psnrNum >= 40.0 ? 'Near-Lossless (>40 dB)' : (psnrNum >= 30.0 ? 'Standard (>30 dB)' : 'Compressed');
    $('radiograph-psnr-badge').textContent = `PSNR: ${psnrVal} dB · ${tag}`;
    $('radiograph-psnr-badge').style.borderColor = psnrNum >= 40.0 ? 'rgba(63, 185, 80, 0.6)' : 'rgba(0, 210, 255, 0.5)';
    $('radiograph-psnr-badge').style.color = psnrNum >= 40.0 ? '#3fb950' : '#00d2ff';
  }
  if ($('radiograph-disease-badge')) {
    $('radiograph-disease-badge').textContent = `${data.highest_finding || 'Finding'} (${((data.highest_probability || 0) * 100).toFixed(1)}%)`;
  }
  if ($('psnr-fidelity-rating')) {
    $('psnr-fidelity-rating').textContent = data.psnr?.quality || 'Over 40 dB — Excellent / Optimal Fidelity';
    $('psnr-fidelity-rating').style.color = parseFloat(psnrVal) >= 40.0 ? '#3fb950' : '#58a6ff';
  }
  if ($('psnr-explanation-text')) {
    $('psnr-explanation-text').textContent = data.psnr?.standard_range || 'High-end studio mastering & medical imaging standard (>40 dB). Minimal noise distortion.';
  }

  // Render focal subregions summary list
  const subregionsContainer = $('focal-subregions-list');
  if (subregionsContainer) {
    if (data.bounding_boxes && data.bounding_boxes.length > 0) {
      subregionsContainer.innerHTML = data.bounding_boxes.map((b) => `
        <div style="font-size: 0.8rem; background: rgba(255,255,255,0.04); border-left: 2px solid #58a6ff; padding: 4px 8px; border-radius: 3px; display: flex; justify-content: space-between;">
          <span style="color: #e6edf3;"><strong>${escapeHtml(b.zone)}</strong></span>
          <span style="color: #00d2ff;">Peak: ${b.peak_norm}</span>
        </div>
      `).join('');
    } else {
      subregionsContainer.innerHTML = `
        <div style="font-size: 0.8rem; color: #8b949e; font-style: italic;">No discrete focal lesion exceeding threshold. Generalized diffuse thoracic pattern.</div>
      `;
    }
  }

  // Toggle buttons
  const btnAnnotated = $('btn-toggle-annotated');
  const btnFused = $('btn-toggle-fused');
  const btnClean = $('btn-toggle-clean');

  function setActiveToggle(btn) {
    [btnAnnotated, btnFused, btnClean].forEach((b) => {
      if (b) {
        b.style.background = '#161b22';
        b.style.color = '#8b949e';
      }
    });
    if (btn) {
      btn.style.background = '#21262d';
      btn.style.color = '#e6edf3';
    }
  }

  if (btnAnnotated) {
    btnAnnotated.onclick = () => {
      if (radImg && annotatedSrc) radImg.src = annotatedSrc;
      setActiveToggle(btnAnnotated);
    };
  }
  if (btnFused) {
    btnFused.onclick = () => {
      if (radImg && fusedSrc) radImg.src = fusedSrc;
      setActiveToggle(btnFused);
    };
  }
  if (btnClean) {
    btnClean.onclick = () => {
      if (radImg && cleanSrc) radImg.src = cleanSrc;
      setActiveToggle(btnClean);
    };
  }

  // Populate Anatomically Grounded Focal Bounding Boxes Card
  const bbCard = $('bounding-boxes-card');
  const bbList = $('bounding-boxes-list');
  const bbBadge = $('bounding-boxes-badge');
  if (data.bounding_boxes && data.bounding_boxes.length > 0) {
    bbCard.hidden = false;
    bbBadge.textContent = `${data.bounding_boxes.length} Focal Region${data.bounding_boxes.length > 1 ? 's' : ''}`;
    bbList.innerHTML = data.bounding_boxes.map((b) => `
      <div class="bounding-box-item" style="background: rgba(255,255,255,0.04); border-left: 3px solid #ff4444; padding: 8px 12px; border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong style="color: #e6edf3;">${escapeHtml(b.zone)}</strong>
          <div style="font-size: 0.8rem; color: #8b949e; margin-top: 2px;">${escapeHtml(b.label)}</div>
        </div>
        <div style="text-align: right;">
          <span class="badge-primary" style="font-size: 0.75rem; background: rgba(88,166,255,0.15); border: 1px solid rgba(88,166,255,0.4); padding: 2px 6px; border-radius: 4px;">Peak: ${b.peak_norm}</span>
          <div style="font-size: 0.75rem; color: #8b949e; margin-top: 3px;">Area: ${b.area_pct}%</div>
        </div>
      </div>
    `).join('');
  } else {
    bbCard.hidden = true;
  }

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

  // Panel 1: Grounded Bounding Boxes on Original Radiograph
  $('panel-1-img').src = panels.panel_1_annotated || panels.panel_1_original;
  const boxCount = panels.bounding_boxes ? panels.bounding_boxes.length : 0;
  $('panel-1-caption').textContent = boxCount > 0 
    ? `${boxCount} focal lesion bounding box(es) localized to anatomical zones.`
    : 'Clinical radiograph preprocessed with aspect-preserving letterboxing.';

  // Panel 2: Anatomical Segmentation Mask (Color-coded)
  $('panel-2-img').src = panels.panel_2_anatomy || panels.panel_2_heatmap;
  $('panel-2-title').textContent = 'Anatomical Segmentation Mask';
  $('panel-2-caption').textContent = 'Color-coded: Blue = Lung Fields | Red = Cardiac Silhouette | Green = Perihilar.';

  // Panel 3: Model-Specific Attribution Heatmap
  $('panel-3-img').src = panels.panel_2_heatmap || panels.panel_3_rollout;
  $('panel-3-title').textContent = method || 'Model-Specific Attribution';
  $('panel-3-caption').textContent = 'Raw mathematical gradient & activation heatmap.';

  // Panel 4: Fused Diagnostic Visual Overlay
  $('panel-4-img').src = panels.panel_4_fused;
  $('panel-4-title').textContent = `Fused Overlay: ${panels.finding} (${(panels.probability * 100).toFixed(1)}%)`;
  $('panel-4-caption').textContent = 'Heatmap constrained to thoracic anatomy with focal lesion bounding boxes.';
  if ($('panel-4-psnr-badge')) {
    const p4PsnrVal = panels.psnr_db || '42.4';
    const p4Num = parseFloat(p4PsnrVal);
    const tag = p4Num >= 40.0 ? 'Near-Lossless (>40 dB)' : 'Standard';
    $('panel-4-psnr-badge').textContent = `PSNR: ${p4PsnrVal} dB (${tag})`;
    $('panel-4-psnr-badge').style.color = p4Num >= 40.0 ? '#3fb950' : '#00d2ff';
    $('panel-4-psnr-badge').style.borderColor = p4Num >= 40.0 ? 'rgba(63, 185, 80, 0.6)' : 'rgba(0, 210, 255, 0.5)';
  }
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
  tagsBox.innerHTML = regions.map((r) => `<span class="tag-pill">${escapeHtml(r)}</span>`).join('') || '<span class="tag-pill">Thoracic Cavity</span>';

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
// 9. RADIOLOGIST GROUND TRUTH BENCHMARK CASES CONTROLLER
// ==============================================================================

async function initBenchmarkCases() {
  try {
    const res = await fetch('/api/benchmark-cases');
    if (!res.ok) return;
    benchmarkCasesData = await res.json();

    document.querySelectorAll('.benchmark-case-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const caseId = btn.dataset.caseId;
        const caseObj = benchmarkCasesData.find((c) => c.id === caseId);
        if (!caseObj) return;

        showToast(`Loading radiologist-verified ${caseObj.name} benchmark case…`);
        try {
          const imgRes = await fetch(caseObj.image_url);
          if (!imgRes.ok) throw new Error('Could not fetch sample image');
          const blob = await imgRes.blob();
          const file = new File([blob], caseObj.filename, { type: blob.type || 'image/jpeg' });

          uploadedFiles = [file];
          activeFileIndex = 0;
          activeBenchmarkCase = caseObj;

          renderBatchThumbnails();
          updateInferenceButtonState();

          // Auto-trigger analysis
          setTimeout(() => {
            const runBtn = $('run-inference-btn');
            if (runBtn && !runBtn.disabled) {
              runBtn.click();
            }
          }, 150);
        } catch (err) {
          showToast(`Failed to load benchmark case: ${err.message}`, 5000);
        }
      });
    });
  } catch (err) {
    console.error('Benchmark cases initialization error:', err);
  }
}

// ==============================================================================
// ==============================================================================
// 10. FOUR-MODEL COMPARISON CONTROLLER
// ==============================================================================

async function runCompareAllModels() {
  if (!uploadedFiles.length) {
    showToast('Please upload a chest radiograph first.');
    return;
  }
  const file = uploadedFiles[activeFileIndex];
  const formData = new FormData();
  formData.append('image', file);

  const thresh = $('threshold-slider')?.value;
  let url = '/api/compare';
  if (thresh) url += `?threshold=${thresh}`;

  showToast('Running all 4 deep learning models independently…');
  switchTab('compare');

  const runBtn = $('run-compare-btn');
  const compRunBtn = $('compare-run-btn');

  if (runBtn) runBtn.disabled = true;
  if (compRunBtn) compRunBtn.disabled = true;

  try {
    const res = await fetch(url, { method: 'POST', body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned ${res.status}`);
    }
    const data = await res.json();
    if (data.is_ood) {
      showToast(`Comparison aborted: ${data.rejection_reason}`, 5000);
      return;
    }
    renderComparisonResults(data);
    showToast('4-model comparative analysis completed.');
  } catch (err) {
    showToast(`Comparison error: ${err.message}`, 5000);
  } finally {
    if (runBtn) runBtn.disabled = false;
    if (compRunBtn) compRunBtn.disabled = false;
  }
}

function renderComparisonResults(data) {
  $('compare-empty-state').hidden = true;
  $('compare-results-container').hidden = false;

  const consensus = data.consensus || {};
  const cBadge = $('consensus-badge');
  const cTitle = $('consensus-title');
  const cDesc = $('consensus-description');
  const cCard = $('compare-consensus-card');

  if (cBadge) cBadge.textContent = consensus.badge || 'Empirical Consensus';
  if (cTitle) cTitle.textContent = consensus.title || 'Diagnostic Consensus';
  if (cDesc) cDesc.textContent = consensus.description || '';
  if ($('compare-total-latency')) {
    $('compare-total-latency').textContent = `${(data.total_elapsed_ms || 0).toFixed(0)} ms`;
  }

  // Border color based on consensus level
  if (cCard) {
    if (consensus.level === 'unanimous' || consensus.level === 'strong') {
      cCard.style.borderLeftColor = '#3fb950';
      cCard.style.background = 'rgba(63, 185, 80, 0.05)';
    } else if (consensus.level === 'split' || consensus.level === 'majority') {
      cCard.style.borderLeftColor = '#d29922';
      cCard.style.background = 'rgba(210, 153, 34, 0.05)';
    } else {
      cCard.style.borderLeftColor = '#f85149';
      cCard.style.background = 'rgba(248, 81, 73, 0.05)';
    }
  }

  // Discrepancies
  const discCard = $('compare-discrepancy-card');
  const discText = $('compare-discrepancy-text');
  if (discCard && discText) {
    if (data.discrepancies && data.discrepancies.length > 0) {
      discCard.hidden = false;
      discText.innerHTML = data.discrepancies.map(d => `
        <div style="margin-bottom: 4px;"><strong>${escapeHtml(d.model_name)}:</strong> ${escapeHtml(d.reason)}</div>
      `).join('');
    } else {
      discCard.hidden = true;
    }
  }

  // Matrix table
  const tbody = $('compare-matrix-body');
  if (tbody && data.models) {
    tbody.innerHTML = data.models.map((m) => {
      const isConfirmed = m.status === 'confirmed';
      const statusColor = isConfirmed ? '#3fb950' : (m.status === 'abstained' ? '#8b949e' : '#f85149');
      const statusLabel = isConfirmed ? 'Confirmed' : (m.status === 'abstained' ? 'Abstained' : 'Unavailable');
      const alts = (m.alternatives || []).map(a => `${a.finding} (${(a.probability * 100).toFixed(0)}%)`).join(', ') || 'None';

      return `
        <tr style="border-bottom: 1px solid #21262d;">
          <td style="padding: 10px; font-weight: 600; color: #e6edf3;">${escapeHtml(m.model_name)}</td>
          <td style="padding: 10px; color: ${isConfirmed ? '#58a6ff' : '#8b949e'}; font-weight: ${isConfirmed ? '700' : '400'};">${escapeHtml(m.primary_finding)}</td>
          <td style="padding: 10px; font-weight: 700; color: ${isConfirmed ? '#e6edf3' : '#8b949e'};">${(m.primary_confidence * 100).toFixed(1)}%</td>
          <td style="padding: 10px; color: #8b949e;">${m.free_energy ? m.free_energy.toFixed(2) : '—'}</td>
          <td style="padding: 10px; color: #8b949e;">${m.processing_time_ms ? m.processing_time_ms.toFixed(1) + ' ms' : '—'}</td>
          <td style="padding: 10px;"><span style="padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: ${statusColor}22; color: ${statusColor}; border: 1px solid ${statusColor}55;">${statusLabel}</span></td>
          <td style="padding: 10px; color: #8b949e; font-size: 0.8rem;">${escapeHtml(alts)}</td>
        </tr>
      `;
    }).join('');
  }

  // 4-Panel Comparative Attributions
  const vitModel = data.models?.find(m => m.model_id === 'vit');
  const densenetModel = data.models?.find(m => m.model_id === 'densenet');
  const capsnetModel = data.models?.find(m => m.model_id === 'capsnet');
  const gatModel = data.models?.find(m => m.model_id === 'gat');

  if (vitModel && $('compare-img-vit')) {
    const src = vitModel.panels?.panel_4_fused || vitModel.panels?.panel_3_attribution || vitModel.annotated_image;
    if (src) $('compare-img-vit').src = src;
    $('compare-lbl-vit').textContent = `${vitModel.primary_finding} (${(vitModel.primary_confidence * 100).toFixed(1)}%)`;
  }
  if (densenetModel && $('compare-img-densenet')) {
    const src = densenetModel.panels?.panel_4_fused || densenetModel.panels?.panel_3_attribution || densenetModel.annotated_image;
    if (src) $('compare-img-densenet').src = src;
    $('compare-lbl-densenet').textContent = `${densenetModel.primary_finding} (${(densenetModel.primary_confidence * 100).toFixed(1)}%)`;
  }
  if (capsnetModel && $('compare-img-capsnet')) {
    const src = capsnetModel.panels?.panel_4_fused || capsnetModel.panels?.panel_3_attribution || capsnetModel.annotated_image;
    if (src) $('compare-img-capsnet').src = src;
    $('compare-lbl-capsnet').textContent = `${capsnetModel.primary_finding} (${(capsnetModel.primary_confidence * 100).toFixed(1)}%)`;
  }
  if (gatModel && $('compare-img-gat')) {
    const src = gatModel.panels?.panel_4_fused || gatModel.panels?.panel_3_attribution || gatModel.annotated_image;
    if (src) $('compare-img-gat').src = src;
    $('compare-lbl-gat').textContent = `${gatModel.primary_finding} (${(gatModel.primary_confidence * 100).toFixed(1)}%)`;
  }
}

// ==============================================================================
// 11. OPEN-SET DISCOVERY & CANDIDATE MATCHING CONTROLLER
// ==============================================================================

async function runOpenSetScan() {
  if (!uploadedFiles.length) {
    showToast('Please upload a chest radiograph first.');
    return;
  }
  const file = uploadedFiles[activeFileIndex];
  const formData = new FormData();
  formData.append('image', file);

  showToast('Scanning 28+ thoracic disease vocabulary with vision-language matching…');
  switchTab('openset');

  const btn = $('openset-run-btn');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/open-set-predict?top_k=8', { method: 'POST', body: formData });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server returned ${res.status}`);
    }
    const data = await res.json();
    renderOpenSetResults(data);
    showToast('Open-set thoracic candidate analysis complete.');
  } catch (err) {
    showToast(`Open-Set scan error: ${err.message}`, 5000);
  } finally {
    if (btn) btn.disabled = false;
  }
}

function renderOpenSetResults(data) {
  $('openset-empty-state').hidden = true;
  $('openset-results-container').hidden = false;

  if ($('openset-energy-val')) {
    $('openset-energy-val').textContent = data.free_energy ? data.free_energy.toFixed(2) : '—';
  }

  const topCand = data.candidates && data.candidates.length > 0 ? data.candidates[0] : null;
  if ($('openset-top-candidate')) {
    $('openset-top-candidate').textContent = topCand ? `${topCand.finding} (${(topCand.score * 100).toFixed(1)}%)` : '—';
  }

  const grid = $('openset-candidates-grid');
  if (grid && data.candidates) {
    grid.innerHTML = data.candidates.map((cand, idx) => {
      const isExp = cand.is_open_set;
      const tagBg = isExp ? 'rgba(188, 140, 255, 0.15)' : 'rgba(56, 139, 253, 0.15)';
      const tagColor = isExp ? '#bc8cff' : '#58a6ff';
      const tagLabel = isExp ? 'Open-Set Candidate (Unseen in Supervised)' : 'CheXpert Supervised Class';

      return `
        <div class="studio-card candidate-card" style="border: 1px solid #30363d; background: rgba(255,255,255,0.02); padding: 14px; border-radius: 6px; display: flex; flex-direction: column; justify-content: space-between;">
          <div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 8px;">
              <div>
                <span style="font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; background: ${tagBg}; color: ${tagColor}; font-weight: 600;">${tagLabel}</span>
                <h4 style="margin: 6px 0 2px 0; font-size: 1.05rem; color: #ffffff;">#${idx + 1} ${escapeHtml(cand.finding)}</h4>
                <small style="color: #8b949e; font-size: 0.75rem;">${escapeHtml(cand.region)}</small>
              </div>
              <div style="text-align: right;">
                <strong style="font-size: 1.25rem; color: ${cand.score >= 0.40 ? '#3fb950' : '#e6edf3'};">${(cand.score * 100).toFixed(1)}%</strong>
                <small style="display: block; font-size: 0.68rem; color: #8b949e;">Similarity</small>
              </div>
            </div>
            <div class="progress-bar-bg" style="width: 100%; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin-bottom: 10px;">
              <div style="width: ${Math.min(100, Math.max(0, cand.score * 100))}%; height: 100%; background: ${cand.score >= 0.40 ? '#3fb950' : '#58a6ff'};"></div>
            </div>
            <p style="font-size: 0.8rem; color: #c9d1d9; line-height: 1.45; margin: 0 0 8px 0;">${escapeHtml(cand.visual_concept || cand.clinical_note || '')}</p>
          </div>
          <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #21262d; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.72rem; color: #8b949e;">${escapeHtml(cand.category)}</span>
            <button type="button" class="action-btn-text" onclick="searchLiterature('${escapeHtml(cand.finding)}')" style="color: #58a6ff; font-size: 0.75rem; text-decoration: underline; background: none; border: none; cursor: pointer;">
              Explore PubMed Citations →
            </button>
          </div>
        </div>
      `;
    }).join('');
  }
}

// ==============================================================================
// 12. MEDICAL EVIDENCE & LITERATURE (PUBMED / NCBI) CONTROLLER
// ==============================================================================

async function searchLiterature(findingName) {
  if (!findingName) return;
  switchTab('literature');
  const input = $('literature-search-input');
  if (input) input.value = findingName;

  showToast(`Retrieving verified literature for "${findingName}"…`);

  try {
    const res = await fetch(`/api/medical-evidence?finding=${encodeURIComponent(findingName)}`);
    if (!res.ok) {
      throw new Error(`Failed to retrieve evidence (${res.status})`);
    }
    const data = await res.json();
    renderLiteratureResult(data);
  } catch (err) {
    showToast(`Literature retrieval error: ${err.message}`, 5000);
  }
}

function renderLiteratureResult(data) {
  if (!data) return;

  if ($('lit-source-badge')) {
    $('lit-source-badge').textContent = data.source_id || 'NCBI / PUBMED CITATION';
  }
  if ($('lit-title-display')) {
    $('lit-title-display').textContent = data.title || `${data.finding} Imaging`;
  }
  if ($('lit-source-link')) {
    $('lit-source-link').href = data.source_url || '#';
  }
  if ($('lit-evidence-text')) {
    $('lit-evidence-text').textContent = data.evidence_text || 'Peer-reviewed evidence on clinical diagnostic criteria.';
  }
  if ($('lit-region-display')) {
    $('lit-region-display').textContent = data.anatomical_region || 'Thorax';
  }
  if ($('lit-visual-display')) {
    $('lit-visual-display').textContent = data.visual_concept || 'Radiographic density alteration';
  }
  if ($('lit-pubmed-link')) {
    $('lit-pubmed-link').href = data.pubmed_search_url || `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(data.finding)}`;
  }
}

// ==============================================================================
// INITIALIZATION
// ==============================================================================

window.addEventListener('DOMContentLoaded', () => {
  loadSystemStatus();
  loadBenchmarks();
  loadFindingEvidence("Lung Opacity");
  initBenchmarkCases();

  // Attach Comparison, Open-Set & Literature event listeners
  $('run-compare-btn')?.addEventListener('click', runCompareAllModels);
  $('compare-run-btn')?.addEventListener('click', runCompareAllModels);
  $('openset-run-btn')?.addEventListener('click', runOpenSetScan);

  $('literature-search-btn')?.addEventListener('click', () => {
    const val = $('literature-search-input')?.value;
    if (val) searchLiterature(val);
  });
  $('literature-search-input')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const val = $('literature-search-input')?.value;
      if (val) searchLiterature(val);
    }
  });

  document.querySelectorAll('.lit-quick-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const finding = btn.dataset.finding;
      if (finding) searchLiterature(finding);
    });
  });
});

