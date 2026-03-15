/**
 * NIDS Dashboard — Minimal Single-Page App
 * Tabs: Dashboard | Analysis | Predict
 */

const COLORS = { indigo:'#6366f1', violet:'#8b5cf6', cyan:'#06b6d4', green:'#22c55e', amber:'#f59e0b' };
const PALETTE = [COLORS.indigo, COLORS.violet, COLORS.cyan, COLORS.green, COLORS.amber];
const MODEL_ORDER = ['cuda', 'mpi', 'openmp', 'thread', 'pyspark'];
const MODEL_ICONS = { cuda:'⚡', mpi:'🔗', openmp:'🧵', thread:'🪡', pyspark:'🔥' };
const LABEL_MAP = {0:'DDoS', 1:'DoS', 2:'NormalTraffic', 3:'PortScan'};

let models = {};
const charts = {};

// ── Bootstrap ──
document.addEventListener('DOMContentLoaded', () => { setupTabs(); init(); });

// ── Tabs ──
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
        });
    });
}

async function init() {
    try {
        const [modelsRes, overview] = await Promise.all([api('/api/models'), api('/api/overview')]);
        models = modelsRes.models;
        setStatus(true);
        renderDashboard(overview);
        loadAnalysis();
        setupPredict();
    } catch (e) { console.error(e); setStatus(false); }
}

async function api(path) { const r = await fetch(path); if (!r.ok) throw new Error(r.status); return r.json(); }
function setStatus(ok) {
    document.getElementById('status').innerHTML = ok
        ? '<span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span><span>Connected</span>'
        : '<span class="w-1.5 h-1.5 rounded-full bg-red-500"></span><span>Offline</span>';
}

// ══════════════ DASHBOARD TAB ══════════════
function renderDashboard(ov) { renderKPI(ov); renderModelCards(); renderTable(); renderCharts(); renderDetails(); }

function renderKPI(ov) {
    const kpis = [
        { label:'Best Accuracy', value:ov.best_model.accuracy.toFixed(1)+'%', sub:ov.best_model.name },
        { label:'Avg Accuracy', value:ov.avg_accuracy.toFixed(1)+'%', sub:`${ov.total_models} models` },
        { label:'Fastest', value:fmt(ov.fastest_model.total_ms), sub:ov.fastest_model.name },
        { label:'Peak GFLOPS', value:(ov.highest_gflops.gflops||0).toFixed(1), sub:ov.highest_gflops.name },
    ];
    document.getElementById('kpi-strip').innerHTML = kpis.map(k => `
        <div class="card px-5 py-4">
            <p class="text-[11px] text-dim uppercase tracking-wider mb-1">${k.label}</p>
            <p class="text-2xl font-bold tracking-tight mono">${k.value}</p>
            <p class="text-xs text-subtle mt-1">${k.sub}</p>
        </div>`).join('');
}

function renderModelCards() {
    document.getElementById('model-cards').innerHTML = ordered().map(([id,m],i) => `
        <div class="card p-4 hover:border-zinc-700 transition-colors cursor-default">
            <div class="flex items-center gap-2 mb-3">
                <span class="text-lg">${MODEL_ICONS[id]||'📊'}</span>
                <span class="text-sm font-semibold">${m.name}</span>
            </div>
            <div class="space-y-2 text-xs">
                <div class="flex justify-between"><span class="text-dim">Accuracy</span><span class="mono font-medium text-zinc-200">${m.accuracy.toFixed(2)}%</span></div>
                <div class="w-full bg-zinc-800 rounded-full h-1.5"><div class="h-1.5 rounded-full" style="width:${m.accuracy}%;background:${PALETTE[i]}"></div></div>
                <div class="flex justify-between"><span class="text-dim">Total</span><span class="mono text-zinc-300">${fmt(m.total_ms)}</span></div>
                <div class="flex justify-between"><span class="text-dim">GFLOPS</span><span class="mono text-zinc-300">${(m.gflops||0).toFixed(1)}</span></div>
            </div>
        </div>`).join('');
}

function renderTable() {
    document.getElementById('perf-tbody').innerHTML = ordered().map(([id,m],i) => `
        <tr class="border-b border-zinc-800/50 hover:bg-surface-2/50 transition-colors">
            <td class="px-5 py-3"><div class="flex items-center gap-2"><span class="w-2 h-2 rounded-full" style="background:${PALETTE[i]}"></span><span class="font-medium text-zinc-200">${m.name}</span><span class="text-[10px] text-dim hidden sm:inline">${m.technique||''}</span></div></td>
            <td class="px-5 py-3 mono text-zinc-200 font-medium">${m.accuracy.toFixed(2)}%</td>
            <td class="px-5 py-3 mono text-zinc-400">${fmt(m.train_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-400">${fmt(m.predict_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-400">${fmt(m.dbscan_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-300">${fmt(m.total_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-300">${(m.gflops||0).toFixed(2)}</td>
        </tr>`).join('');
}

const chartDef = {
    responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
    scales:{ x:{ticks:{color:'#71717a',font:{size:11}},grid:{display:false}}, y:{ticks:{color:'#52525b',font:{size:10}},grid:{color:'#1f1f23'}} }
};

function renderCharts() {
    const names = ordered().map(([,m])=>m.name);
    mk('chart-acc','bar',names,[{data:ordered().map(([,m])=>m.accuracy),backgroundColor:PALETTE,borderRadius:4,barThickness:32}],
        {...chartDef,scales:{...chartDef.scales,y:{...chartDef.scales.y,min:Math.max(0,Math.min(...ordered().map(([,m])=>m.accuracy))-3),max:100}}});
    mk('chart-time','bar',names,[{data:ordered().map(([,m])=>m.total_ms/1000),backgroundColor:PALETTE,borderRadius:4,barThickness:32}],
        {...chartDef,scales:{...chartDef.scales,y:{...chartDef.scales.y,ticks:{...chartDef.scales.y.ticks,callback:v=>v+'s'}}}});
    mk('chart-pipeline','bar',names,[
        {label:'Train',data:ordered().map(([,m])=>m.train_ms/1000),backgroundColor:COLORS.indigo+'cc',borderRadius:3},
        {label:'Predict',data:ordered().map(([,m])=>m.predict_ms/1000),backgroundColor:COLORS.violet+'cc',borderRadius:3},
        {label:'DBSCAN',data:ordered().map(([,m])=>m.dbscan_ms/1000),backgroundColor:COLORS.cyan+'cc',borderRadius:3},
    ],{...chartDef,plugins:{legend:{labels:{color:'#71717a',font:{size:10},usePointStyle:true,pointStyleWidth:6,padding:16}}},
        scales:{x:{stacked:true,...chartDef.scales.x},y:{stacked:true,...chartDef.scales.y,ticks:{...chartDef.scales.y.ticks,callback:v=>v+'s'}}}});
    mk('chart-gflops','bar',names,[{data:ordered().map(([,m])=>m.gflops||0),backgroundColor:PALETTE,borderRadius:4,barThickness:28}],
        {...chartDef,indexAxis:'y',scales:{x:{...chartDef.scales.y},y:{ticks:{color:'#a1a1aa',font:{size:11,weight:500}},grid:{display:false}}}});
}

function renderDetails() {
    document.getElementById('model-details').innerHTML = ordered().map(([id,m],i) => `
        <details class="card group">
            <summary class="px-5 py-3.5 cursor-pointer flex items-center justify-between hover:bg-surface-2/50 transition-colors rounded-xl list-none">
                <div class="flex items-center gap-3"><span class="w-2.5 h-2.5 rounded-full" style="background:${PALETTE[i]}"></span><span class="text-sm font-semibold">${m.name}</span><span class="text-[11px] text-dim">${m.technique||''}</span></div>
                <svg class="w-4 h-4 text-dim transition-transform group-open:rotate-180" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
            </summary>
            <div class="px-5 pb-5 pt-2 border-t border-zinc-800/60">
                <p class="text-sm text-subtle mb-4">${m.description||''}</p>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-10 gap-y-1 text-xs">
                    ${dr('Parallelism',m.parallelism||'-')}${dr('Hardware',m.hardware||'-')}
                    ${dr('Train Samples',num(m.n_train))}${dr('Test Samples',num(m.n_test))}
                    ${dr('Features',m.n_features)}${dr('Classes',m.n_classes)}
                    ${dr('SVM Training',fmt(m.train_ms))}${dr('SVM Prediction',fmt(m.predict_ms))}
                    ${dr('DBSCAN',fmt(m.dbscan_ms))}${dr('Total Time',fmt(m.total_ms))}
                    ${dr('GFLOPS',(m.gflops||0).toFixed(2))}${dr('FLOP Count',num(m.flops||0))}
                    ${m.confident?dr('Confident / Uncertain',`${num(m.confident)} / ${num(m.uncertain)}`):''}
                    ${m.dbscan_clusters?dr('DBSCAN Clusters / Noise',`${m.dbscan_clusters} / ${m.dbscan_noise}`):''}
                </div>
            </div>
        </details>`).join('');
}

// ══════════════ ANALYSIS TAB ══════════════
async function loadAnalysis() {
    try {
        const analysis = await api('/api/analysis');
        renderAnalysisCharts(analysis);
        renderAnalysisTables(analysis);
        renderConfusionMatrices();
    } catch(e) { console.error('Analysis load failed:', e); }
}

function renderAnalysisCharts(analysis) {
    // Grouped bar: Accuracy, Precision, Recall, F1 per model
    const names = ordered().map(([,m])=>m.name);
    const accuracies = ordered().map(([,m])=>m.accuracy);

    // Get the per-class macro metrics from analysis (use the best available source per model)
    const analysisKeys = {
        cuda:'cuda_hybrid', mpi:'mpi', openmp:'openmp', thread:'thread', pyspark:'pyspark'
    };
    const precisions = [], recalls = [], f1s = [];
    for (const [id] of ordered()) {
        const key = analysisKeys[id];
        const data = analysis[key];
        if (data && data.metrics) {
            const macro = data.metrics.find(m => m.class === 'macro avg' || m.class === 'macro');
            if (macro) {
                precisions.push(macro.precision * 100);
                recalls.push(macro.recall * 100);
                f1s.push(macro.f1 * 100);
                continue;
            }
        }
        // Fallback: use accuracy for all
        const acc = models[id]?.accuracy || 0;
        precisions.push(acc);
        recalls.push(acc);
        f1s.push(acc);
    }

    mk('chart-analysis-metrics','bar',names,[
        {label:'Accuracy',data:accuracies,backgroundColor:'#6366f1cc',borderRadius:3,barPercentage:0.8,categoryPercentage:0.7},
        {label:'Precision',data:precisions,backgroundColor:'#8b5cf6cc',borderRadius:3,barPercentage:0.8,categoryPercentage:0.7},
        {label:'Recall',data:recalls,backgroundColor:'#06b6d4cc',borderRadius:3,barPercentage:0.8,categoryPercentage:0.7},
        {label:'F1',data:f1s,backgroundColor:'#22c55ecc',borderRadius:3,barPercentage:0.8,categoryPercentage:0.7},
    ],{...chartDef,plugins:{legend:{labels:{color:'#a1a1aa',font:{size:10},usePointStyle:true,pointStyleWidth:6,padding:16}}},
        scales:{...chartDef.scales,y:{...chartDef.scales.y,min:Math.max(0,Math.min(...accuracies,...precisions,...recalls,...f1s)-5),max:100}}});

    // Training time
    mk('chart-analysis-train','bar',names,[{data:ordered().map(([,m])=>m.train_ms/1000),backgroundColor:PALETTE,borderRadius:4,barThickness:32}],
        {...chartDef,scales:{...chartDef.scales,y:{...chartDef.scales.y,ticks:{...chartDef.scales.y.ticks,callback:v=>v.toFixed(0)+'s'}}}});
}

function renderAnalysisTables(analysis) {
    const el = document.getElementById('analysis-tables');
    let html = '';
    for (const [key, data] of Object.entries(analysis)) {
        if (!data.metrics || data.metrics.length === 0) continue;
        html += `<div class="card p-4">
            <h4 class="text-xs font-semibold text-zinc-300 mb-3">${key.replace(/_/g,' ').toUpperCase()} <span class="text-dim font-normal">(${data.source})</span></h4>
            <table class="w-full text-xs"><thead><tr class="border-b border-zinc-800">
                <th class="py-1.5 text-left text-dim">Class</th><th class="py-1.5 text-right text-dim">Prec</th><th class="py-1.5 text-right text-dim">Recall</th><th class="py-1.5 text-right text-dim">F1</th><th class="py-1.5 text-right text-dim">Support</th>
            </tr></thead><tbody>
            ${data.metrics.map(m => `<tr class="border-b border-zinc-800/30"><td class="py-1.5 text-zinc-300">${m.class}</td><td class="py-1.5 text-right mono text-zinc-400">${(m.precision*100).toFixed(1)}%</td><td class="py-1.5 text-right mono text-zinc-400">${(m.recall*100).toFixed(1)}%</td><td class="py-1.5 text-right mono text-zinc-300">${(m.f1*100).toFixed(1)}%</td><td class="py-1.5 text-right mono text-dim">${num(m.support)}</td></tr>`).join('')}
            </tbody></table></div>`;
    }
    el.innerHTML = html || '<p class="text-sm text-dim">No per-class metrics available from log files.</p>';
}

function renderConfusionMatrices() {
    // Build confusion matrices from prediction files vs ground truth
    // Since we don't have ground truth CSVs in the frontend, show prediction distribution as proxy
    const el = document.getElementById('confusion-matrices');
    const modelNames = ordered().map(([id,m])=>({id,name:m.name}));

    Promise.all(modelNames.map(m => api(`/api/models/${m.id}/predictions`).then(d => ({...m, ...d})).catch(()=>null)))
        .then(results => {
            const valid = results.filter(r => r && r.total > 0);
            el.innerHTML = valid.map((r, idx) => {
                const dist = r.distribution;
                const keys = Object.keys(dist).sort((a,b) => parseInt(a)-parseInt(b));
                const total = r.total;
                return `<div class="card p-4">
                    <h4 class="text-xs font-semibold text-zinc-300 mb-3">${r.name}</h4>
                    <table class="cm-table">
                        <thead><tr><th>Class</th><th>Count</th><th>%</th><th>Bar</th></tr></thead>
                        <tbody>
                        ${keys.map((k,i) => {
                            const cnt = dist[k];
                            const pct = (cnt/total*100).toFixed(1);
                            const label = LABEL_MAP[parseInt(k)] || (k=='-1'?'Anomaly/Noise':`Class ${k}`);
                            const maxCnt = Math.max(...Object.values(dist));
                            return `<tr><td class="text-left text-zinc-300 text-[11px]">${label}</td>
                                <td class="cm-cell">${num(cnt)}</td><td class="cm-cell">${pct}%</td>
                                <td><div class="h-1.5 rounded-full" style="width:${(cnt/maxCnt*100).toFixed(0)}%;background:${PALETTE[i%5]}"></div></td></tr>`;
                        }).join('')}
                        </tbody>
                        <tfoot><tr><td class="text-left font-medium text-zinc-200 text-[11px]">Total</td><td class="cm-cell font-medium text-zinc-200">${num(total)}</td><td></td><td></td></tr></tfoot>
                    </table>
                </div>`;
            }).join('');
        });
}

// ══════════════ PREDICT TAB ══════════════
let selectedFile = null;
let selectedModel = 'pyspark';
let predictModels = [];

function setupPredict() {
    loadPredictModels();

    document.getElementById('btn-predict-manual').addEventListener('click', predictManual);
    document.getElementById('btn-load-sample').addEventListener('click', loadSampleData);
    document.getElementById('btn-load-attacks').addEventListener('click', loadAttackSamples);

    const dropZone = document.getElementById('drop-zone');
    const csvInput = document.getElementById('csv-upload');

    dropZone.addEventListener('click', () => csvInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('border-indigo-500'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-indigo-500'));
    dropZone.addEventListener('drop', e => {
        e.preventDefault(); dropZone.classList.remove('border-indigo-500');
        if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    csvInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

    document.getElementById('btn-predict-csv').addEventListener('click', predictCSV);
}

function handleFile(file) {
    selectedFile = file;
    document.getElementById('csv-filename').textContent = file.name;
    document.getElementById('btn-predict-csv').disabled = false;
}

async function loadPredictModels() {
    try {
        const data = await api('/api/predict/models');
        predictModels = data.models;
        renderModelSelector();
        loadPredictInfo(selectedModel);
    } catch(e) { console.error(e); }
}

function renderModelSelector() {
    const container = document.getElementById('model-selector');
    if (!container) return;

    container.innerHTML = predictModels.map((m, i) => `
        <button class="model-select-btn ${m.id === selectedModel ? 'active' : ''}" data-model="${m.id}">
            <span class="text-lg">${MODEL_ICONS[m.id] || '📊'}</span>
            <div class="flex flex-col items-start">
                <span class="font-medium text-xs">${m.name}</span>
                <span class="text-[10px] text-dim">${m.accuracy.toFixed(1)}% acc</span>
            </div>
            <span class="w-2 h-2 rounded-full ${m.loaded ? 'bg-emerald-500' : 'bg-amber-500'}"></span>
        </button>
    `).join('');

    container.querySelectorAll('.model-select-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            selectedModel = btn.dataset.model;
            container.querySelectorAll('.model-select-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadPredictInfo(selectedModel);
        });
    });
}

async function loadPredictInfo(modelId = 'pyspark') {
    try {
        const info = await api(`/api/predict/info?model=${modelId}`);
        const el = document.getElementById('predict-model-info');
        el.innerHTML = `
            <div class="flex items-center gap-2 mb-3">
                <span class="w-2 h-2 rounded-full ${info.model_loaded?'bg-emerald-500':'bg-red-500'}"></span>
                <span class="text-sm font-medium">${info.model_loaded?'Model Ready':'Model Not Found'}</span>
                <span class="text-xs text-dim ml-auto">${info.technique || ''}</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
                ${dr2('Type',info.model_type)}${dr2('Features',info.n_features)}
                ${dr2('Sub-models',info.n_sub_models)}${dr2('Gamma',info.gamma?.toFixed(4)||'-')}
            </div>
            <div class="mt-3 text-xs">
                <span class="text-dim">Classes:</span> ${Object.entries(info.classes||{}).map(([k,v])=>`<span class="text-zinc-400">${v}</span>`).join(' • ')}
            </div>
            ${info.accuracy > 0 ? `<div class="mt-2 text-xs"><span class="text-dim">Test Accuracy:</span> <span class="text-emerald-400 font-medium">${info.accuracy.toFixed(2)}%</span></div>` : ''}
            <p class="mt-2 text-xs text-dim">${info.description || ''}</p>`;
    } catch(e) { console.error(e); }
}

// Pre-defined attack samples from CICIDS dataset (52 features each)
const ATTACK_SAMPLES = {
    ddos: [0.01, 0.85, 0.0001, 0, 0.0005, 0.002, 0.003, 0.001, 0, 0.0003, 0.002, 0.001, 0, 0.112, 0.35, 0.0001, 0, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.999, 0.994, 0.018, 0.026, 0.004, 0.0002, 0.002, 0, 0, 0, 0, 1, 0.002, 0.0005, 0.004, 0.004, 0, 1, 0, 0, 0, 0, 0],
    dos: [0, 0.0003, 0.1, 0.0001, 0.0002, 0.026, 0, 0.016, 0.02, 0.05, 0, 0.015, 0.027, 0.112, 0.333, 0.002, 0.008, 0.02, 0, 0.09, 0.005, 0.012, 0.02, 0, 0.1, 0.004, 0.01, 0.02, 0, 0.999, 0.994, 0, 0, 0, 0.039, 0.026, 0.04, 0.002, 0, 1, 0, 0.023, 0.0002, 0.446, 0.004, 0.0001, 1, 0, 0, 0, 0, 0],
    portscan: [0, 0.0003, 0.07, 0.00004, 0.00001, 0.001, 0, 0.002, 0.001, 0.002, 0, 0.002, 0.002, 0.112, 0.333, 0.003, 0.012, 0.027, 0, 0.05, 0.006, 0.016, 0.028, 0, 0.07, 0.005, 0.014, 0.027, 0, 0.999, 0.994, 0, 0, 0, 0.001, 0.004, 0.003, 0.00001, 0, 1, 0, 0.003, 0.00001, 0.446, 0.003, 0.00003, 1, 0, 0, 0, 0, 0],
    normal: [0, 0.12, 0.0008, 0.00001, 0.00002, 0.008, 0, 0.009, 0.014, 0.007, 0, 0.008, 0.009, 0.112, 0.333, 0.0001, 0.0004, 0.0007, 0, 0.0008, 0.0003, 0.0006, 0.0007, 0, 0.0007, 0.0004, 0.0007, 0.0007, 0, 0.999, 0.994, 0.00002, 0.00002, 0, 0.008, 0.013, 0.016, 0.0003, 0, 1, 0, 0.013, 0.00002, 0.125, 0.004, 0.00001, 1, 0, 0, 0, 0, 0],
};

function loadSampleData() {
    // Show a menu to pick attack type
    const types = ['ddos', 'dos', 'portscan', 'normal'];
    const currentIdx = (window._sampleIdx || 0) % types.length;
    const type = types[currentIdx];
    window._sampleIdx = currentIdx + 1;

    const sample = ATTACK_SAMPLES[type].map(v => v.toFixed(6));
    document.getElementById('manual-input').value = `# ${type.toUpperCase()} sample\n` + sample.join(', ');
}

function loadAttackSamples() {
    // Load multiple attack samples
    const samples = [
        '# DDoS Attack',
        ATTACK_SAMPLES.ddos.map(v => v.toFixed(6)).join(', '),
        '# DoS Attack',
        ATTACK_SAMPLES.dos.map(v => v.toFixed(6)).join(', '),
        '# Port Scan',
        ATTACK_SAMPLES.portscan.map(v => v.toFixed(6)).join(', '),
    ];
    document.getElementById('manual-input').value = samples.join('\n');
}

async function predictManual() {
    const text = document.getElementById('manual-input').value.trim();
    if (!text) return;

    // Filter out comment lines (starting with #) and empty lines
    const lines = text.split('\n').filter(l => l.trim() && !l.trim().startsWith('#'));
    const features = lines.map(l => l.split(',').map(v=>parseFloat(v.trim())).filter(v=>!isNaN(v)));

    if (features.length === 0) {
        alert('No valid data lines found');
        return;
    }

    if (features.some(f => f.length < 52)) {
        alert(`Each line needs 52 features. Got ${features.map(f=>f.length).join(', ')}`);
        return;
    }

    try {
        const res = await fetch('/api/predict', {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({features: features.map(f=>f.slice(0,52)), model: selectedModel})
        });
        const data = await res.json();
        if (!res.ok) { alert(data.detail); return; }
        showPredictResults(data);
    } catch(e) { alert('Prediction failed: '+e.message); }
}

async function predictCSV() {
    if (!selectedFile) return;
    const form = new FormData();
    form.append('file', selectedFile);
    form.append('model', selectedModel);
    try {
        const res = await fetch('/api/predict/csv', {method:'POST', body:form});
        const data = await res.json();
        if (!res.ok) { alert(data.detail); return; }
        showPredictResults(data);
    } catch(e) { alert('CSV prediction failed: '+e.message); }
}

function showPredictResults(data) {
    const section = document.getElementById('predict-results');
    section.classList.remove('hidden');

    // Update header with model info
    const header = section.querySelector('h3.section-title');
    if (header) {
        header.innerHTML = `Prediction Results <span class="text-indigo-400 ml-2">${data.model || selectedModel}</span>`;
    }

    // Build summary from predictions
    const counts = {};
    if (data.predictions) {
        data.predictions.forEach(p => { counts[p.predicted_label] = (counts[p.predicted_label]||0)+1; });
    }
    const summaryData = data.summary || counts;
    const classCounts = Object.entries(summaryData);

    const summaryHtml = classCounts.length > 0 ? classCounts.map(([cls,cnt]) => {
        const isNormal = cls === 'NormalTraffic';
        const color = isNormal ? 'text-emerald-400' : 'text-amber-400';
        return `<div class="card px-4 py-3">
            <p class="text-[11px] text-dim uppercase tracking-wider mb-1">${cls}</p>
            <p class="text-xl font-bold mono ${color}">${cnt}</p>
        </div>`;
    }).join('') : `
        <div class="card px-4 py-3 col-span-2">
            <p class="text-[11px] text-dim uppercase tracking-wider mb-1">Total Predictions</p>
            <p class="text-xl font-bold mono">${data.n_samples}</p>
        </div>`;

    document.getElementById('predict-summary').innerHTML = summaryHtml;

    // Results table (limit to 100 rows)
    const preds = data.predictions || [];
    const show = preds.slice(0, 100);
    document.getElementById('predict-results-tbody').innerHTML = show.map((p,i) => {
        const voteStr = Object.entries(p.votes||{}).map(([c,v])=>`${c}:${v}`).join('  ');
        const labelColor = p.predicted_label==='NormalTraffic'?'text-emerald-400':'text-amber-400';
        return `<tr class="border-b border-zinc-800/30 hover:bg-surface-2/50">
            <td class="px-5 py-2 text-dim">${i+1}</td>
            <td class="px-5 py-2 font-medium ${labelColor}">${p.predicted_label}</td>
            <td class="px-5 py-2 mono text-zinc-400">${p.confidence.toFixed(4)}</td>
            <td class="px-5 py-2 mono text-xs text-dim">${voteStr}</td>
        </tr>`;
    }).join('') + (preds.length>100 ? `<tr><td colspan="4" class="px-5 py-3 text-center text-dim text-xs">... and ${preds.length-100} more</td></tr>` : '');

    section.scrollIntoView({behavior:'smooth', block:'start'});
}

// ══════════════ HELPERS ══════════════
function ordered() { return MODEL_ORDER.filter(id=>models[id]).map(id=>[id,models[id]]); }
function fmt(ms) { if(!ms||ms===0)return'—'; if(ms<1000)return ms.toFixed(0)+' ms'; if(ms<60000)return(ms/1000).toFixed(1)+' s'; return(ms/60000).toFixed(1)+' min'; }
function num(n) { if(n==null)return'—'; return Number(n).toLocaleString(); }
function mk(id,type,labels,datasets,options) { if(charts[id])charts[id].destroy(); charts[id]=new Chart(document.getElementById(id),{type,data:{labels,datasets},options}); }
function dr(l,v) { return `<div class="flex justify-between py-1.5 border-b border-zinc-800/30"><span class="text-dim">${l}</span><span class="mono text-zinc-300">${v}</span></div>`; }
function dr2(l,v) { return `<div><span class="text-dim">${l}:</span> <span class="mono text-zinc-300">${v}</span></div>`; }
