const $ = (id) => document.getElementById(id);
let selectedFile = null;
let models = [];
let lastPrediction = null;

function toast(message){const el=$('toast');el.textContent=message;el.classList.add('show');clearTimeout(toast.timer);toast.timer=setTimeout(()=>el.classList.remove('show'),4200);}
function escapeHtml(value){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function switchTab(name){document.querySelectorAll('.tab').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));document.querySelectorAll('.tab-panel').forEach(p=>p.classList.toggle('active',p.id===`panel-${name}`));if(name==='comparison')loadBenchmarks();}
document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>switchTab(b.dataset.tab)));
$('threshold').addEventListener('input',()=>{$('threshold-value').textContent=Number($('threshold').value).toFixed(2);if(lastPrediction)renderPredictions(lastPrediction);});

function setSelectedFile(file){
  if(!file)return;
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){toast('Please choose a JPG, PNG or WEBP image.');return;}
  if(file.size>20*1024*1024){toast('Image is larger than 20 MB.');return;}
  selectedFile=file;
  const url=URL.createObjectURL(file);$('image-preview').src=url;$('image-preview').hidden=false;
  $('image-info').textContent=`${file.name} · ${(file.size/1024).toFixed(0)} KB`;
  $('clear-image').disabled=false;updateAnalyzeButton();
}
$('image-file').addEventListener('change',e=>setSelectedFile(e.target.files[0]));
$('clear-image').addEventListener('click',()=>{selectedFile=null;$('image-file').value='';$('image-preview').hidden=true;$('image-preview').removeAttribute('src');$('image-info').textContent='No image selected';$('clear-image').disabled=true;updateAnalyzeButton();});
const dz=$('dropzone');['dragenter','dragover'].forEach(evt=>dz.addEventListener(evt,e=>{e.preventDefault();dz.classList.add('dragover');}));['dragleave','drop'].forEach(evt=>dz.addEventListener(evt,e=>{e.preventDefault();dz.classList.remove('dragover');}));dz.addEventListener('drop',e=>setSelectedFile(e.dataTransfer.files[0]));
$('model-select').addEventListener('change',updateAnalyzeButton);
function updateAnalyzeButton(){const model=models.find(m=>m.id===$('model-select').value);const ready=!!model?.available;$('analyze-button').disabled=!selectedFile||!ready;$('analyze-hint').textContent=!selectedFile?'Upload an image to continue.':!ready?'The selected architecture has no trained checkpoint. Train it in Colab, then copy the .pt file into checkpoints/.':'Ready to run inference using the selected trained checkpoint.';$('model-hint').textContent=model?model.reason:'No models are registered.';}

async function loadStatus(){
 try{
  const [healthRes,modelRes]=await Promise.all([fetch('/api/health'),fetch('/api/models')]);
  if(!healthRes.ok||!modelRes.ok)throw new Error('API status request failed');
  const health=await healthRes.json();models=(await modelRes.json()).models;
  $('runtime-device').textContent=`${health.device.toUpperCase()} · ${health.models_with_checkpoints}/${health.models_total} checkpoints`;
  $('api-status').textContent='API online';$('api-status').classList.remove('muted');
  $('model-select').innerHTML=models.map(m=>`<option value="${escapeHtml(m.id)}" ${m.available?'':'disabled'}>${escapeHtml(m.name)} · ${m.available?'READY':'CHECKPOINT MISSING'}</option>`).join('');
  const first=models.find(m=>m.available);if(first)$('model-select').value=first.id;
  updateAnalyzeButton();
 }catch(err){$('api-status').textContent='API unavailable';$('runtime-device').textContent='Not connected';$('model-select').innerHTML='<option>Backend API unavailable</option>';$('analyze-button').disabled=true;$('analyze-hint').textContent='Start the FastAPI server, then refresh this page.';}
}

function renderPredictions(data){
 const threshold=Number($('threshold').value);
 $('result-model').textContent=data.model_name;$('result-time').textContent=`${data.processing_time_ms.toFixed(1)} ms`;$('result-threshold').textContent=threshold.toFixed(2);
 const findings=data.findings.map(f=>({...f,positive:f.probability>=threshold}));
 $('probability-list').innerHTML=findings.sort((a,b)=>b.probability-a.probability).map(f=>`<div class="prob-row ${f.positive?'positive':''}"><span class="prob-label">${escapeHtml(f.finding)}</span><span class="bar-track"><span class="bar-fill" style="display:block;width:${Math.max(0,Math.min(100,f.probability*100))}%"></span></span><span class="prob-value">${(f.probability*100).toFixed(1)}%</span></div>`).join('');
 $('empty-results').hidden=true;$('results-content').hidden=false;$('result-status').textContent=`${findings.filter(f=>f.positive).length} above threshold`;
 const graph=data.graph_context;
 const relations=(graph?.relations||[]).slice(0,14).map(r=>`<div class="graph-rel">${escapeHtml(r.source)} <b>→ ${escapeHtml(r.relation)} →</b> ${escapeHtml(r.target)}</div>`).join('');
 const ev=(graph?.evidence||[]).map(e=>`<div class="graph-rel">${escapeHtml(e.title||e.finding)} · <a href="${escapeHtml(e.source_url)}" target="_blank" rel="noopener">Source</a></div>`).join('');
 $('graph-context').innerHTML=`<p>${escapeHtml(graph?.message||'No graph context available.')}</p>${relations?`<div class="graph-relations">${relations}</div>`:''}${ev}`;
 if(data.explanation?.available&&data.explanation.overlay){$('xai-empty').hidden=true;$('xai-content').hidden=false;$('xai-original').src=$('image-preview').src;$('xai-overlay').src=data.explanation.overlay;$('xai-method').textContent=data.explanation.method||'Model-derived attribution';}else{$('xai-empty').hidden=false;$('xai-content').hidden=true;$('xai-empty').textContent=data.explanation?.reason||'No explanation was returned by the model.';}
}

$('analyze-button').addEventListener('click',async()=>{
 if(!selectedFile)return;const model=$('model-select').value;const threshold=$('threshold').value;
 $('analyze-button').disabled=true;$('analyze-button').innerHTML='<span>Analyzing…</span><span class="spinner">◌</span>';$('result-status').textContent='Running inference';
 try{
  const form=new FormData();form.append('image',selectedFile);
  const res=await fetch(`/api/predict?model=${encodeURIComponent(model)}&threshold=${threshold}&explain=true`,{method:'POST',body:form});
  const body=await res.json();if(!res.ok)throw new Error(body.detail||'Prediction failed');
  lastPrediction=body;renderPredictions(body);toast('Inference completed using the selected trained model.');
 }catch(err){$('result-status').textContent='Prediction unavailable';toast(err.message);}
 finally{$('analyze-button').innerHTML='<span>Analyze Chest X-ray</span><span>↗</span>';updateAnalyzeButton();}
});

async function loadBenchmarks(){
 $('benchmark-rows').innerHTML='<tr><td colspan="7" class="table-empty">Loading benchmark register…</td></tr>';
 try{
  const res=await fetch('/api/benchmarks');if(!res.ok)throw new Error('Cannot load benchmarks');const data=await res.json();
  $('benchmark-rows').innerHTML=data.benchmarks.map(b=>{const m=b.metrics||{};const fmt=x=>Number.isFinite(Number(x))?Number(x).toFixed(4):'—';return `<tr><td><strong>${escapeHtml(b.name)}</strong></td><td>${escapeHtml(b.status)}</td><td>${fmt(m.macro_auroc)}</td><td>${fmt(m.macro_auprc)}</td><td>${fmt(m.micro_f1)}</td><td>${fmt(m.macro_f1)}</td><td>${m.validation_samples??'—'}</td></tr>`;}).join('');
  $('benchmark-note').textContent=data.note;
 }catch(err){$('benchmark-rows').innerHTML='<tr><td colspan="7" class="table-empty">Benchmark API unavailable.</td></tr>';}
}
$('refresh-benchmarks').addEventListener('click',loadBenchmarks);
loadStatus();loadBenchmarks();
