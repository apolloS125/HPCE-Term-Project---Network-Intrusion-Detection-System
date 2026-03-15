/**
 * NIDS Dashboard — Advanced Glassmorphic App
 * Tabs: Dashboard | Analysis | Predict
 */

const COLORS = { 
    indigo: '#6366f1', 
    violet: '#8b5cf6', 
    cyan: '#06b6d4', 
    green: '#10b981', 
    amber: '#f59e0b',
    fuchsia: '#d946ef'
};
const PALETTE = [COLORS.indigo, COLORS.violet, COLORS.cyan, COLORS.green, COLORS.amber, COLORS.fuchsia];
const MODEL_ORDER = ['cuda', 'mpi', 'openmp', 'thread', 'pyspark'];
const MODEL_ICONS = { cuda:'⚡', mpi:'🔗', openmp:'🧵', thread:'🪡', pyspark:'🔥' };
const LABEL_MAP = {0:'DDoS', 1:'DoS', 2:'NormalTraffic', 3:'PortScan'};

let models = {};
const charts = {};

// ── Bootstrap ──
document.addEventListener('DOMContentLoaded', () => { 
    Chart.defaults.font.family = 'Inter, sans-serif';
    Chart.defaults.color = '#71717a';
    setupTabs(); 
    init(); 
});

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
    const el = document.getElementById('status');
    if (ok) {
        el.innerHTML = `<span class="relative flex h-2 w-2 mr-1">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
        </span> <span class="text-emerald-500 font-medium">System Online</span>`;
        el.className = "flex items-center gap-1.5 text-xs text-emerald-500 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded-full";
    } else {
        el.innerHTML = `<span class="h-2 w-2 rounded-full bg-red-500 mr-1"></span> Offline`;
        el.className = "flex items-center gap-1.5 text-xs text-red-500 bg-red-500/10 border border-red-500/20 px-2.5 py-1 rounded-full";
    }
}

// ══════════════ DASHBOARD TAB ══════════════
function renderDashboard(ov) { renderKPI(ov); renderModelCards(); renderTable(); renderCharts(); renderDetails(); }

function renderKPI(ov) {
    const kpis = [
        { label:'Peak Accuracy', value:ov.best_model.accuracy.toFixed(1)+'%', sub:ov.best_model.name },
        { label:'Cluster Avg Accuracy', value:ov.avg_accuracy.toFixed(1)+'%', sub:`Across ${ov.total_models} models` },
        { label:'Lowest Latency', value:fmt(ov.fastest_model.total_ms), sub:ov.fastest_model.name },
        { label:'Max Throughput', value:(ov.highest_gflops.gflops||0).toFixed(1) + ' GF', sub:ov.highest_gflops.name },
    ];
    document.getElementById('kpi-strip').innerHTML = kpis.map(k => `
        <div class="card p-6 flex flex-col justify-center relative overflow-hidden group hoverable">
            <div class="absolute -right-6 -top-6 w-24 h-24 bg-indigo-500/10 rounded-full blur-2xl group-hover:bg-indigo-500/20 transition-colors"></div>
            <p class="text-[10px] text-dim uppercase tracking-widest font-display font-medium mb-1.5">${k.label}</p>
            <p class="text-3xl font-bold tracking-tight kpi-value mb-1">${k.value}</p>
            <p class="text-xs text-indigo-400 font-medium">${k.sub}</p>
        </div>`).join('');
}

function renderModelCards() {
    document.getElementById('model-cards').innerHTML = ordered().map(([id,m],i) => `
        <div class="card p-5 hoverable cursor-default flex flex-col">
            <div class="flex items-center gap-3 mb-4">
                <div class="w-8 h-8 rounded-lg bg-white/5 border border-white/10 flex items-center justify-center text-lg shadow-inner">
                    ${MODEL_ICONS[id]||'📊'}
                </div>
                <span class="text-[15px] font-display font-semibold text-white tracking-wide">${m.name}</span>
            </div>
            <div class="space-y-3 box-border">
                <div class="space-y-1.5">
                    <div class="flex justify-between items-end">
                        <span class="text-[11px] text-dim uppercase tracking-wider font-medium">Accuracy</span>
                        <span class="font-mono text-xs font-semibold text-white">${m.accuracy.toFixed(2)}%</span>
                    </div>
                    <div class="w-full bg-black/40 rounded-full h-1.5 shadow-inner overflow-hidden border border-white/5">
                        <div class="h-full rounded-full" style="width:${m.accuracy}%; background:${PALETTE[i]}; box-shadow: 0 0 10px ${PALETTE[i]}"></div>
                    </div>
                </div>
                <div class="h-px w-full bg-white/5 my-2"></div>
                <div class="flex justify-between items-center text-xs">
                    <span class="text-dim">Execution Time</span>
                    <span class="font-mono text-zinc-300">${fmt(m.total_ms)}</span>
                </div>
                <div class="flex justify-between items-center text-xs">
                    <span class="text-dim">GFLOPS</span>
                    <span class="font-mono text-zinc-300">${(m.gflops||0).toFixed(1)}</span>
                </div>
            </div>
        </div>`).join('');
}

function renderTable() {
    document.getElementById('perf-tbody').innerHTML = ordered().map(([id,m],i) => `
        <tr class="border-b border-white/5 hover:bg-white/[0.03] transition-colors group">
            <td class="px-5 py-4">
                <div class="flex items-center gap-3">
                    <div class="w-2.5 h-2.5 rounded-full shadow-[0_0_8px_${PALETTE[i]}]" style="background:${PALETTE[i]}"></div>
                    <div class="flex flex-col">
                        <span class="font-display font-medium text-zinc-100 group-hover:text-white transition-colors">${m.name}</span>
                        <span class="text-[10px] text-dim">${m.technique||''}</span>
                    </div>
                </div>
            </td>
            <td class="px-5 py-4 font-mono text-[13px] text-zinc-100 font-medium">${m.accuracy.toFixed(2)}%</td>
            <td class="px-5 py-4 font-mono text-[13px] text-zinc-400">${fmt(m.train_ms)}</td>
            <td class="px-5 py-4 font-mono text-[13px] text-zinc-400">${fmt(m.predict_ms)}</td>
            <td class="px-5 py-4 font-mono text-[13px] text-zinc-400">${fmt(m.dbscan_ms)}</td>
            <td class="px-5 py-4 font-mono text-[13px] text-zinc-300 font-medium">${fmt(m.total_ms)}</td>
            <td class="px-5 py-4 font-mono text-[13px] text-zinc-300">${(m.gflops||0).toFixed(2)}</td>
        </tr>`).join('');
}

function createGradient(ctx, color) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, color + '20'); // transparent version
    return gradient;
}

const chartDef = {
    responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}},
    scales:{ 
        x:{ticks:{color:'#71717a',font:{family:'Outfit',size:12}},grid:{display:false},border:{display:false}}, 
        y:{ticks:{color:'#71717a',font:{family:'JetBrains Mono',size:11}},grid:{color:'rgba(255,255,255,0.05)',drawBorder:false},border:{display:false}} 
    }
};

function renderCharts() {
    const names = ordered().map(([,m])=>m.name);
    
    // Custom logic to add gradients once canvas ctx is available
    const mkGrad = id => {
        const ctx = document.getElementById(id).getContext('2d');
        return PALETTE.map(c => createGradient(ctx, c));
    };

    setTimeout(() => {
        mk('chart-acc','bar',names,[{data:ordered().map(([,m])=>m.accuracy),backgroundColor:mkGrad('chart-acc'),borderRadius:6,borderWidth:1,borderColor:PALETTE,barThickness:36}],
            {...chartDef,scales:{...chartDef.scales,y:{...chartDef.scales.y,min:Math.max(0,Math.min(...ordered().map(([,m])=>m.accuracy))-3),max:100}}});
        
        mk('chart-time','bar',names,[{data:ordered().map(([,m])=>m.total_ms/1000),backgroundColor:mkGrad('chart-time'),borderRadius:6,borderWidth:1,borderColor:PALETTE,barThickness:36}],
            {...chartDef,scales:{...chartDef.scales,y:{...chartDef.scales.y,ticks:{...chartDef.scales.y.ticks,callback:v=>v+'s'}}}});
            
        mk('chart-pipeline','bar',names,[
            {label:'Train',data:ordered().map(([,m])=>m.train_ms/1000),backgroundColor:COLORS.indigo+'99',borderRadius:4,borderWidth:1,borderColor:COLORS.indigo},
            {label:'Predict',data:ordered().map(([,m])=>m.predict_ms/1000),backgroundColor:COLORS.violet+'99',borderRadius:4,borderWidth:1,borderColor:COLORS.violet},
            {label:'DBSCAN',data:ordered().map(([,m])=>m.dbscan_ms/1000),backgroundColor:COLORS.cyan+'99',borderRadius:4,borderWidth:1,borderColor:COLORS.cyan},
        ],{...chartDef,plugins:{legend:{labels:{color:'#a1a1aa',font:{family:'Outfit',size:11},usePointStyle:true,pointStyleWidth:8,padding:20}}},
            scales:{x:{stacked:true,...chartDef.scales.x},y:{stacked:true,...chartDef.scales.y,ticks:{...chartDef.scales.y.ticks,callback:v=>v+'s'}}}});
            
        mk('chart-gflops','bar',names,[{data:ordered().map(([,m])=>m.gflops||0),backgroundColor:mkGrad('chart-gflops'),borderRadius:6,borderWidth:1,borderColor:PALETTE,barThickness:32}],
            {...chartDef,indexAxis:'y',scales:{x:{...chartDef.scales.y},y:{ticks:{color:'#a1a1aa',font:{family:'Outfit',size:12,weight:500}},grid:{display:false}}}});
    }, 50);
}

function renderDetails() {
    document.getElementById('model-details').innerHTML = ordered().map(([id,m],i) => `
        <details class="card group transition-all duration-300">
            <summary class="px-6 py-4 cursor-pointer flex items-center justify-between hover:bg-white/[0.02] transition-colors rounded-xl list-none focus:outline-none focus:ring-1 focus:ring-indigo-500/50">
                <div class="flex items-center gap-4">
                    <div class="w-3 h-3 rounded-full shadow-[0_0_8px_${PALETTE[i]}]" style="background:${PALETTE[i]}"></div>
                    <span class="font-display text-base font-semibold text-white tracking-wide">${m.name}</span>
                    <span class="text-[11px] font-mono text-zinc-400 bg-white/5 px-2 py-0.5 rounded hidden sm:inline-block border border-white/5">${m.technique||''}</span>
                </div>
                <svg class="w-5 h-5 text-zinc-500 transition-transform duration-300 group-open:rotate-180" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 9l-7 7-7-7"/></svg>
            </summary>
            <div class="px-6 pb-6 pt-3 border-t border-white/5 bg-black/20">
                <p class="text-sm text-dim mb-6 leading-relaxed">${m.description||''}</p>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-12 gap-y-2 text-[13px]">
                    ${dr('Architecture',m.parallelism||'-')}${dr('Hardware',m.hardware||'-')}
                    ${dr('Training Corpus',num(m.n_train))}${dr('Test Corpus',num(m.n_test))}
                    ${dr('Feature Space',m.n_features)}${dr('Output Classes',m.n_classes)}
                    ${dr('SVM Training Time',fmt(m.train_ms))}${dr('SVM Inference Time',fmt(m.predict_ms))}
                    ${dr('DBSCAN Refinement',fmt(m.dbscan_ms))}${dr('Total Execution Time',fmt(m.total_ms))}
                    ${dr('Computational Yield',(m.gflops||0).toFixed(2) + ' GFLOPS', true)}${dr('Total FLOP Count',num(m.flops||0))}
                    ${m.confident?dr('Confident / Uncertain SVs',`<span class="text-emerald-400">${num(m.confident)}</span> / <span class="text-amber-400">${num(m.uncertain)}</span>`):''}
                    ${m.dbscan_clusters?dr('DBSCAN Clusters / Noise',`<span class="text-cyan-400">${m.dbscan_clusters}</span> / <span class="text-fuchsia-400">${m.dbscan_noise}</span>`):''}
                </div>
            </div>
        </details>`).join('');
}

// ══════════════ ANALYSIS TAB ══════════════
async function loadAnalysis() {
    renderComparisons();
    try {
        const analysis = await api('/api/analysis');
        renderAnalysisTables(analysis);
    } catch(e) {
        console.error('Analysis load failed:', e);
        document.getElementById('analysis-tables').innerHTML = '<p class="text-sm text-dim">Granular metrics stream disrupted.</p>';
    }
}

function renderComparisons() {
    if (!models || Object.keys(models).length === 0) return;

    const omp = models.openmp || {}, thr = models.thread || {}, mpi = models.mpi || {}, pys = models.pyspark || {}, cuda = models.cuda || {};

    const ompGflops = omp.gflops || 0, thrGflops = thr.gflops || 0;
    const ompTime = omp.total_ms || 0, thrTime = thr.total_ms || 0;
    const sharedWinner = thrGflops > ompGflops ? 'C++ Thread' : 'OpenMP';
    const sharedSpeedup = ompTime && thrTime ? (Math.max(ompTime, thrTime) / Math.min(ompTime, thrTime)).toFixed(2) : '—';
    
    document.getElementById('cmp-shared-metrics').innerHTML = `
        ${cmpRow('OpenMP Throughput', ompGflops.toFixed(2) + ' GF')}
        ${cmpRow('C++ Thread Throughput', thrGflops.toFixed(2) + ' GF')}
        ${cmpRow('OpenMP Latency', fmt(ompTime))}
        ${cmpRow('C++ Thread Latency', fmt(thrTime))}
        ${cmpRow('Relative Speedup', sharedSpeedup + 'x', true, COLORS.indigo)}
        ${cmpRow('Optimal Choice', sharedWinner, true, COLORS.indigo)}
    `;

    const mpiTime = mpi.total_ms || 0, pysTime = pys.total_ms || 0;
    const mpiAcc = mpi.accuracy || 0, pysAcc = pys.accuracy || 0;
    const distWinner = pysTime && mpiTime ? (pysTime < mpiTime ? 'PySpark' : 'MPI') : '—';
    const distSpeedup = mpiTime && pysTime ? (Math.max(mpiTime, pysTime) / Math.min(mpiTime, pysTime)).toFixed(2) : '—';
    
    document.getElementById('cmp-dist-metrics').innerHTML = `
        ${cmpRow('MPI Latency', fmt(mpiTime))}
        ${cmpRow('PySpark Latency', fmt(pysTime))}
        ${cmpRow('MPI Accuracy', mpiAcc.toFixed(2) + '%')}
        ${cmpRow('PySpark Accuracy', pysAcc.toFixed(2) + '%')}
        ${cmpRow('Relative Speedup', distSpeedup + 'x', true, COLORS.violet)}
        ${cmpRow('Performance Lead', distWinner, true, COLORS.violet)}
    `;

    const cudaGflops = cuda.gflops || 0, cudaTime = cuda.total_ms || 0;
    const cudaAcc = cuda.accuracy || 0, ompAcc = omp.accuracy || 0;
    const gpuWinner = cudaGflops > ompGflops ? 'CUDA' : 'OpenMP';
    
    document.getElementById('cmp-gpu-metrics').innerHTML = `
        ${cmpRow('GPU (CUDA) Throughput', cudaGflops.toFixed(2) + ' GF')}
        ${cmpRow('CPU (OpenMP) Throughput', ompGflops.toFixed(2) + ' GF')}
        ${cmpRow('GPU Time', fmt(cudaTime))}
        ${cmpRow('CPU Time', fmt(ompTime))}
        ${cmpRow('GPU Precision', cudaAcc.toFixed(2) + '%')}
        ${cmpRow('Computational Superiority', gpuWinner, true, COLORS.cyan)}
    `;

    const accDiff = (thrAcc => cudaAcc - thrAcc)(thr.accuracy || 0);
    const kernelWinner = cudaAcc > (thr.accuracy||0) ? 'RFF (CUDA)' : 'Exact (Thread)';
    
    document.getElementById('cmp-kernel-metrics').innerHTML = `
        ${cmpRow('RFF Approx Accuracy', cudaAcc.toFixed(2) + '%')}
        ${cmpRow('Exact Kernel Accuracy', (thr.accuracy||0).toFixed(2) + '%')}
        ${cmpRow('RFF Evaluation Time', fmt(cudaTime))}
        ${cmpRow('Exact Evaluation Time', fmt(thrTime))}
        ${cmpRow('Precision Deviation', (accDiff >= 0 ? '+' : '') + accDiff.toFixed(2) + '%', true, accDiff >= 0 ? COLORS.green : COLORS.amber)}
        ${cmpRow('Algorithmic Winner', kernelWinner, true, COLORS.amber)}
    `;

    setTimeout(() => {
        mk('chart-cmp-shared','bar',['OpenMP','Thread'],[{label:'GFLOPS',data:[ompGflops, thrGflops],backgroundColor:[COLORS.indigo+'cc',COLORS.indigo],borderRadius:4,barThickness:36}],{...chartDef,indexAxis:'y'});
        mk('chart-cmp-dist','bar',['MPI','PySpark'],[{label:'Time (s)',data:[mpiTime/1000, pysTime/1000],backgroundColor:[COLORS.violet+'cc',COLORS.violet],borderRadius:4,barThickness:36}],{...chartDef,indexAxis:'y'});
        mk('chart-cmp-gpu','bar',['CUDA','OpenMP'],[{label:'GFLOPS',data:[cudaGflops, ompGflops],backgroundColor:[COLORS.cyan,COLORS.cyan+'66'],borderRadius:4,barThickness:36}],{...chartDef,indexAxis:'y'});
        mk('chart-cmp-kernel','bar',['RFF','Exact'],[{label:'Accuracy',data:[cudaAcc, thr.accuracy||0],backgroundColor:[COLORS.amber,COLORS.amber+'66'],borderRadius:4,barThickness:36}],{...chartDef,indexAxis:'y',scales:{x:{...chartDef.scales.y,min:80,max:100}}});
    }, 50);

    const summaryData = [
        {cmp:'Shared-memory', models:'OpenMP vs Thread', winner:sharedWinner, color:'text-indigo-400 bg-indigo-500/10 border border-indigo-500/20', finding:`${sharedWinner} achieves ${sharedSpeedup}x speedup relative to counterpart.`},
        {cmp:'Distributed', models:'MPI vs PySpark', winner:distWinner, color:'text-violet-400 bg-violet-500/10 border border-violet-500/20', finding:`${distWinner} demonstrates ${distSpeedup}x superior scalability.`},
        {cmp:'Hardware Topology', models:'CUDA vs OpenMP', winner:gpuWinner, color:'text-cyan-400 bg-cyan-500/10 border border-cyan-500/20', finding:`${gpuWinner} delivers orders of magnitude higher FLOPS.`},
        {cmp:'Kernel Strategy', models:'RFF vs Exact', winner:kernelWinner, color:'text-amber-400 bg-amber-500/10 border border-amber-500/20', finding:`${kernelWinner} provides better accuracy (${accDiff>=0?'+':''}${accDiff.toFixed(2)}% margin).`},
    ];
    document.getElementById('summary-tbody').innerHTML = summaryData.map(r => `
        <tr class="border-b border-white/5 hover:bg-white/[0.02] transition-colors">
            <td class="px-6 py-4 text-zinc-200 font-display font-medium tracking-wide">${r.cmp}</td>
            <td class="px-6 py-4 font-mono text-zinc-400">${r.models}</td>
            <td class="px-6 py-4"><span class="px-3 py-1 text-[11px] font-mono tracking-widest rounded shadow-sm ${r.color}">${r.winner}</span></td>
            <td class="px-6 py-4 text-zinc-400 text-sm italic">${r.finding}</td>
        </tr>
    `).join('');
}

function cmpRow(label, value, highlight=false, highlightColor='#10b981') {
    return `<div class="flex justify-between py-2 border-b border-white/5 hover:bg-white/[0.02] transition-colors px-1">
        <span class="text-dim text-xs">${label}</span>
        <span class="font-mono text-xs ${highlight ? 'font-semibold' : 'text-zinc-300'}" style="${highlight ? `color: ${highlightColor}` : ''}">${value}</span>
    </div>`;
}

function renderAnalysisTables(analysis) {
    let html = '';
    for (const [key, data] of Object.entries(analysis)) {
        if (!data.metrics || data.metrics.length === 0) continue;
        html += `<div class="card p-6 border-t-2" style="border-top-color: ${COLORS[key==='cuda'?'cyan':key==='pyspark'?'fuchsia':'indigo']}">
            <h4 class="font-display text-[13px] font-semibold text-white mb-4 uppercase tracking-widest flex justify-between items-center">
                ${key.replace(/_/g,' ')} 
                <span class="text-[10px] text-dim font-mono normal-case tracking-normal border border-white/10 px-2 py-0.5 rounded bg-black/20">${data.source}</span>
            </h4>
            <div class="overflow-x-auto">
                <table class="w-full text-xs">
                    <thead><tr class="border-b border-white/10">
                        <th class="py-2 text-left text-dim font-medium uppercase tracking-wider">Class</th>
                        <th class="py-2 text-right text-dim font-medium uppercase tracking-wider">Precision</th>
                        <th class="py-2 text-right text-dim font-medium uppercase tracking-wider">Recall</th>
                        <th class="py-2 text-right text-dim font-medium uppercase tracking-wider">F1 Score</th>
                        <th class="py-2 text-right text-dim font-medium uppercase tracking-wider">Support</th>
                    </tr></thead>
                    <tbody class="divide-y divide-white/5">
                        ${data.metrics.map(m => `<tr class="hover:bg-white/[0.02] transition-colors">
                            <td class="py-2 text-zinc-200 font-medium">${m.class}</td>
                            <td class="py-2 text-right font-mono text-zinc-400">${(m.precision*100).toFixed(1)}%</td>
                            <td class="py-2 text-right font-mono text-zinc-400">${(m.recall*100).toFixed(1)}%</td>
                            <td class="py-2 text-right font-mono font-medium ${m.f1 > 0.9 ? 'text-emerald-400' : 'text-zinc-200'}">${(m.f1*100).toFixed(1)}%</td>
                            <td class="py-2 text-right font-mono text-dim">${num(m.support)}</td>
                        </tr>`).join('')}
                    </tbody>
                </table>
            </div>
        </div>`;
    }
    document.getElementById('analysis-tables').innerHTML = html || '<p class="text-sm text-dim">No robust granular logs mapped.</p>';
}

// ══════════════ PREDICT TAB ══════════════
let selectedFile = null, selectedModel = 'pyspark', predictModels = [];

function setupPredict() {
    loadPredictModels();
    document.getElementById('btn-predict-manual').addEventListener('click', predictManual);
    document.getElementById('btn-load-sample').addEventListener('click', loadSampleData);
    document.getElementById('btn-load-attacks').addEventListener('click', loadAttackSamples);

    const dropZone = document.getElementById('drop-zone');
    const csvInput = document.getElementById('csv-upload');

    if (!dropZone || !csvInput) {
        console.error('File upload elements not found');
        return;
    }

    dropZone.addEventListener('click', () => {
        console.log('Drop zone clicked');
        csvInput.click();
    });
    dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('border-indigo-500/40', 'bg-indigo-500/5');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-indigo-500/40', 'bg-indigo-500/5');
    });
    dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-500/40', 'bg-indigo-500/5');
        if (e.dataTransfer.files.length) {
            console.log('File dropped:', e.dataTransfer.files[0].name);
            handleFile(e.dataTransfer.files[0]);
        }
    });
    csvInput.addEventListener('change', e => {
        if (e.target.files.length) {
            console.log('File selected:', e.target.files[0].name);
            handleFile(e.target.files[0]);
        }
    });

    document.getElementById('btn-predict-csv').addEventListener('click', predictCSV);
}

function handleFile(file) {
    console.log('handleFile called with:', file);
    if (!file) {
        console.error('No file provided');
        return;
    }
    selectedFile = file;
    const filenameEl = document.getElementById('csv-filename');
    const btnEl = document.getElementById('btn-predict-csv');

    if (filenameEl) filenameEl.textContent = file.name;
    if (btnEl) {
        btnEl.disabled = false;
        console.log('Upload button enabled');
    }
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
    if(!container) return;
    container.innerHTML = predictModels.map((m) => `
        <button class="model-select-btn ${m.id === selectedModel ? 'active' : ''}" data-model="${m.id}">
            <span class="text-xl">${MODEL_ICONS[m.id] || '📊'}</span>
            <div class="flex flex-col items-start pr-2">
                <span class="font-display font-semibold text-sm tracking-wide shadow-sm">${m.name}</span>
                <span class="font-mono text-[10px] text-dim">${m.accuracy.toFixed(1)}% acc</span>
            </div>
            <span class="w-2 h-2 rounded-full ${m.loaded ? 'bg-emerald-500 shadow-[0_0_8px_#10b981]' : 'bg-amber-500 shadow-[0_0_8px_#f59e0b]'}"></span>
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
        el.className = `card p-6 border-l-4 ${info.model_loaded ? 'border-l-indigo-500' : 'border-l-red-500'} transition-colors`;
        el.innerHTML = `
            <div class="flex items-center gap-3 mb-5">
                <span class="relative flex h-3 w-3">
                    ${info.model_loaded ? '<span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>' : ''}
                    <span class="relative inline-flex rounded-full h-3 w-3 ${info.model_loaded?'bg-emerald-500':'bg-red-500'}"></span>
                </span>
                <span class="font-display text-sm font-semibold tracking-wide text-white">${info.model_loaded?'Engine Ready & Armed':'Engine Offline'}</span>
                <span class="text-[10px] font-mono text-dim border border-white/10 px-2 py-0.5 rounded bg-black/20 ml-auto whitespace-nowrap">${info.technique || ''}</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-6 text-[13px]">
                ${dr2('Kernel Strategy',`<span class="text-white">${info.model_type}</span>`)}
                ${dr2('Input Dimensions',info.n_features)}
                ${dr2('Ensemble Branches',info.n_sub_models)}
                ${dr2('Hyperparameter γ',info.gamma?.toFixed(4)||'N/A')}
            </div>
            <div class="h-px bg-white/5 my-4"></div>
            <div class="text-[13px] flex flex-wrap gap-2 items-center">
                <span class="text-dim uppercase tracking-wider text-[10px] font-semibold mr-2">Identifiable Threats:</span> 
                ${Object.entries(info.classes||{}).map(([k,v])=>`<span class="px-2 py-1 bg-white/[0.03] border border-white/5 rounded text-xs font-mono text-zinc-300 shadow-sm">${v}</span>`).join('')}
            </div>
            <p class="mt-4 text-[13px] text-dim leading-relaxed italic border-l-2 border-white/10 pl-3">${info.description || ''}</p>`;
    } catch(e) { console.error(e); }
}

const ATTACK_SAMPLES = {
    ddos: [0.01, 0.85, 0.0001, 0, 0.0005, 0.002, ...Array(46).fill(0).map(()=>Math.random()*0.1)],
    dos: [0, 0.0003, 0.1, 0.0001, 0.0002, 0.026, ...Array(46).fill(0).map(()=>Math.random()*0.1)],
    portscan: [0, 0.0003, 0.07, 0.00004, 0.00001, 0.001, ...Array(46).fill(0).map(()=>Math.random()*0.1)],
    normal: [0, 0.12, 0.0008, 0.00001, 0.00002, 0.008, ...Array(46).fill(0).map(()=>Math.random()*0.01)],
};

function loadSampleData() {
    const types = ['normal'];
    const sample = ATTACK_SAMPLES.normal.map(v => v.toFixed(6));
    document.getElementById('manual-input').value = `# Base Profile: SYN/ACK Normal Flow Topology\n` + sample.join(', ');
}

async function loadAttackSamples() {
    try {
        const response = await fetch('/static/samples.txt');
        const text = await response.text();
        // Extract just the mixed attack batch section
        const lines = text.split('\n');
        const startIdx = lines.findIndex(l => l.includes('Mixed Attack Batch'));
        if (startIdx > 0) {
            const samples = lines.slice(startIdx).join('\n');
            document.getElementById('manual-input').value = samples;
        } else {
            // Fallback to full file
            document.getElementById('manual-input').value = text;
        }
    } catch(e) {
        console.error('Failed to load samples:', e);
        // Fallback to synthetic samples
        const samples = [
            '# Malicious Payload: DDoS Hulk Attempt', ATTACK_SAMPLES.ddos.map(v => v.toFixed(6)).join(', '),
            '# Malicious Payload: DoS Slowloris', ATTACK_SAMPLES.dos.map(v => v.toFixed(6)).join(', '),
            '# Reconnaissance: Intense Port Scan', ATTACK_SAMPLES.portscan.map(v => v.toFixed(6)).join(', ')
        ];
        document.getElementById('manual-input').value = samples.join('\n');
    }
}

async function predictManual() {
    const text = document.getElementById('manual-input').value.trim();
    if (!text) return;

    const lines = text.split('\n').filter(l => l.trim() && !l.trim().startsWith('#'));
    const features = lines.map(l => l.split(',').map(v=>parseFloat(v.trim())).filter(v=>!isNaN(v)));

    if (features.length === 0 || features.some(f => f.length < 52)) {
        alert('Invalid matrix dimension. Ensure 52 floating-point features per flow.'); return;
    }

    try {
        const res = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({features: features.map(f=>f.slice(0,52)), model: selectedModel}) });
        const data = await res.json();
        if (!res.ok) { alert(data.detail); return; }
        showPredictResults(data);
    } catch(e) { alert('Inference pipeline fault: '+e.message); }
}

async function predictCSV() {
    if (!selectedFile) return;
    const form = new FormData(); form.append('file', selectedFile); form.append('model', selectedModel);
    try {
        const res = await fetch('/api/predict/csv', {method:'POST', body:form});
        const data = await res.json();
        if (!res.ok) { alert(data.detail); return; }
        showPredictResults(data);
    } catch(e) { alert('Batch ingestion failed: '+e.message); }
}

function showPredictResults(data) {
    const section = document.getElementById('predict-results');
    section.classList.remove('hidden');
    document.getElementById('predict-results-header').innerHTML = `Inference Telemetry <span class="bg-indigo-500/20 text-indigo-400 font-mono text-[10px] px-2 py-1 rounded ml-3 border border-indigo-500/30 tracking-widest uppercase">${data.model || selectedModel}</span>`;

    const counts = {};
    if (data.predictions) data.predictions.forEach(p => { counts[p.predicted_label] = (counts[p.predicted_label]||0)+1; });
    const classCounts = Object.entries(data.summary || counts);

    document.getElementById('predict-summary').innerHTML = classCounts.length > 0 ? classCounts.map(([cls,cnt]) => {
        const isNormal = cls === 'NormalTraffic';
        const color = isNormal ? 'text-emerald-400' : 'text-amber-400';
        const bgRow = isNormal ? 'bg-emerald-500/10 border-emerald-500/20' : 'bg-amber-500/10 border-amber-500/20';
        return `<div class="card px-5 py-4 border ${bgRow} flex flex-col items-center justify-center relative overflow-hidden">
            <div class="absolute inset-0 bg-gradient-to-t from-black/40 to-transparent"></div>
            <p class="text-[11px] text-dim uppercase tracking-widest font-display font-medium mb-1 relative z-10">${cls}</p>
            <p class="text-3xl font-display font-bold ${color} tracking-tight relative z-10 drop-shadow-md">${cnt}</p>
        </div>`;
    }).join('') : `<div class="card p-5 col-span-2 text-center text-dim">No anomalies detected in payload footprint.</div>`;

    const show = (data.predictions || []).slice(0, 100);
    document.getElementById('predict-results-tbody').innerHTML = show.map((p,i) => {
        const voteStr = Object.entries(p.votes||{}).map(([c,v])=>`<span class="text-zinc-500">${c}:</span><span class="text-zinc-300 ml-1">${v.toFixed(1)}</span>`).join(' <span class="text-white/10 mx-1">|</span> ');
        const isNormal = p.predicted_label === 'NormalTraffic';
        return `<tr class="hover:bg-white/[0.04] transition-colors group">
            <td class="px-5 py-3 text-xs font-mono text-zinc-600 group-hover:text-zinc-400">#${(i+1).toString().padStart(3,'0')}</td>
            <td class="px-5 py-3">
                <span class="inline-flex items-center px-2.5 py-1 rounded border text-xs font-mono font-medium tracking-wide shadow-sm
                    ${isNormal ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-amber-500/10 text-amber-400 border-amber-500/20'}">
                    ${p.predicted_label}
                </span>
            </td>
            <td class="px-5 py-3 font-mono text-[13px] ${p.confidence > 0.8 ? 'text-zinc-100' : 'text-zinc-400'}">${p.confidence.toFixed(4)}</td>
            <td class="px-5 py-3 text-[11px] font-mono whitespace-nowrap overflow-x-auto">${voteStr}</td>
        </tr>`;
    }).join('');
    section.scrollIntoView({behavior:'smooth', block:'start'});
}

// ══════════════ HELPERS ══════════════
function ordered() { return MODEL_ORDER.filter(id=>models[id]).map(id=>[id,models[id]]); }
function fmt(ms) { if(!ms||ms===0)return'—'; if(ms<1000)return ms.toFixed(0)+' ms'; if(ms<60000)return(ms/1000).toFixed(1)+' s'; return(ms/60000).toFixed(1)+' min'; }
function num(n) { if(n==null)return'—'; return Number(n).toLocaleString(); }
function mk(id,type,labels,datasets,options) { if(charts[id])charts[id].destroy(); charts[id]=new Chart(document.getElementById(id),{type,data:{labels,datasets},options}); }
function dr(l,v,hl=false) { return `<div class="flex justify-between py-2 border-b border-white/5 hover:bg-white/[0.02] transition-colors px-1"><span class="text-dim">${l}</span><span class="font-mono ${hl?'font-semibold text-white':'text-zinc-300'}">${v}</span></div>`; }
function dr2(l,v) { return `<div><span class="text-dim block mb-1 uppercase tracking-widest text-[10px] font-medium">${l}</span> <span class="font-mono text-zinc-300 text-sm bg-black/20 border border-white/5 px-2.5 py-1 rounded inline-block w-full">${v}</span></div>`; }
