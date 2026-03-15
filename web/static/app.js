/**
 * NIDS Dashboard — Minimal Single-Page App
 */

const COLORS = {
    indigo: '#6366f1',
    violet: '#8b5cf6',
    cyan:   '#06b6d4',
    green:  '#22c55e',
    amber:  '#f59e0b',
};
const PALETTE = [COLORS.indigo, COLORS.violet, COLORS.cyan, COLORS.green, COLORS.amber];
const MODEL_ORDER = ['cuda', 'mpi', 'openmp', 'thread', 'pyspark'];
const MODEL_ICONS = {
    cuda:    '⚡',
    mpi:     '🔗',
    openmp:  '🧵',
    thread:  '🪡',
    pyspark: '🔥',
};

let models = {};

// ── Bootstrap ──
document.addEventListener('DOMContentLoaded', init);

async function init() {
    try {
        const [modelsRes, overview] = await Promise.all([
            api('/api/models'),
            api('/api/overview'),
        ]);
        models = modelsRes.models;
        setStatus(true);
        render(overview);
    } catch (e) {
        console.error(e);
        setStatus(false);
    }
}

async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(r.status);
    return r.json();
}

function setStatus(ok) {
    const el = document.getElementById('status');
    el.innerHTML = ok
        ? '<span class="w-1.5 h-1.5 rounded-full bg-emerald-500"></span><span>Connected</span>'
        : '<span class="w-1.5 h-1.5 rounded-full bg-red-500"></span><span>Offline</span>';
}

// ── Render ──
function render(overview) {
    renderKPI(overview);
    renderModelCards();
    renderTable();
    renderCharts();
    renderDetails();
}

// ── KPI Strip ──
function renderKPI(ov) {
    const kpis = [
        { label: 'Best Accuracy',  value: ov.best_model.accuracy.toFixed(1) + '%', sub: ov.best_model.name },
        { label: 'Avg Accuracy',   value: ov.avg_accuracy.toFixed(1) + '%',        sub: `${ov.total_models} models` },
        { label: 'Fastest',        value: fmt(ov.fastest_model.total_ms),           sub: ov.fastest_model.name },
        { label: 'Peak GFLOPS',    value: ov.highest_gflops.gflops.toFixed(1),      sub: ov.highest_gflops.name },
    ];
    document.getElementById('kpi-strip').innerHTML = kpis.map(k => `
        <div class="bg-surface-1 border border-zinc-800 rounded-xl px-5 py-4">
            <p class="text-[11px] text-dim uppercase tracking-wider mb-1">${k.label}</p>
            <p class="text-2xl font-bold tracking-tight mono">${k.value}</p>
            <p class="text-xs text-subtle mt-1">${k.sub}</p>
        </div>
    `).join('');
}

// ── Model Cards ──
function renderModelCards() {
    const el = document.getElementById('model-cards');
    el.innerHTML = ordered().map(([id, m], i) => `
        <div class="bg-surface-1 border border-zinc-800 rounded-xl p-4 hover:border-zinc-700 transition-colors cursor-default group">
            <div class="flex items-center gap-2 mb-3">
                <span class="text-lg">${MODEL_ICONS[id] || '📊'}</span>
                <span class="text-sm font-semibold">${m.name}</span>
            </div>
            <div class="space-y-2 text-xs">
                <div class="flex justify-between">
                    <span class="text-dim">Accuracy</span>
                    <span class="mono font-medium text-zinc-200">${m.accuracy.toFixed(2)}%</span>
                </div>
                <div class="w-full bg-zinc-800 rounded-full h-1.5">
                    <div class="h-1.5 rounded-full transition-all duration-700" style="width:${m.accuracy}%;background:${PALETTE[i]}"></div>
                </div>
                <div class="flex justify-between">
                    <span class="text-dim">Total</span>
                    <span class="mono text-zinc-300">${fmt(m.total_ms)}</span>
                </div>
                <div class="flex justify-between">
                    <span class="text-dim">GFLOPS</span>
                    <span class="mono text-zinc-300">${m.gflops.toFixed(1)}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// ── Table ──
function renderTable() {
    const tbody = document.getElementById('perf-tbody');
    tbody.innerHTML = ordered().map(([id, m], i) => `
        <tr class="border-b border-zinc-800/50 hover:bg-surface-2/50 transition-colors">
            <td class="px-5 py-3">
                <div class="flex items-center gap-2">
                    <span class="w-2 h-2 rounded-full" style="background:${PALETTE[i]}"></span>
                    <span class="font-medium text-zinc-200">${m.name}</span>
                    <span class="text-[10px] text-dim hidden sm:inline">${m.technique || ''}</span>
                </div>
            </td>
            <td class="px-5 py-3 mono text-zinc-200 font-medium">${m.accuracy.toFixed(2)}%</td>
            <td class="px-5 py-3 mono text-zinc-400">${fmt(m.train_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-400">${fmt(m.predict_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-400">${fmt(m.dbscan_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-300">${fmt(m.total_ms)}</td>
            <td class="px-5 py-3 mono text-zinc-300">${m.gflops.toFixed(2)}</td>
        </tr>
    `).join('');
}

// ── Charts ──
const charts = {};
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
        x: { ticks: { color: '#71717a', font: { size: 11 } }, grid: { display: false } },
        y: { ticks: { color: '#52525b', font: { size: 10 } }, grid: { color: '#1f1f23' } },
    },
};

function renderCharts() {
    const names = ordered().map(([, m]) => m.name);

    // Accuracy
    makeChart('chart-acc', 'bar', names,
        [{ data: ordered().map(([, m]) => m.accuracy), backgroundColor: PALETTE, borderRadius: 4, barThickness: 32 }],
        { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: Math.max(0, Math.min(...ordered().map(([,m]) => m.accuracy)) - 3), max: 100 }}});

    // Time
    makeChart('chart-time', 'bar', names,
        [{ data: ordered().map(([, m]) => m.total_ms / 1000), backgroundColor: PALETTE, borderRadius: 4, barThickness: 32 }],
        { ...chartDefaults, scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => v + 's' }}}});

    // Pipeline stacked
    makeChart('chart-pipeline', 'bar', names, [
        { label: 'Train',   data: ordered().map(([, m]) => m.train_ms / 1000),   backgroundColor: COLORS.indigo + 'cc', borderRadius: 3 },
        { label: 'Predict', data: ordered().map(([, m]) => m.predict_ms / 1000), backgroundColor: COLORS.violet + 'cc', borderRadius: 3 },
        { label: 'DBSCAN',  data: ordered().map(([, m]) => m.dbscan_ms / 1000),  backgroundColor: COLORS.cyan + 'cc',   borderRadius: 3 },
    ], {
        ...chartDefaults,
        plugins: { legend: { labels: { color: '#71717a', font: { size: 10 }, usePointStyle: true, pointStyleWidth: 6, padding: 16 } } },
        scales: { x: { stacked: true, ...chartDefaults.scales.x }, y: { stacked: true, ...chartDefaults.scales.y, ticks: { ...chartDefaults.scales.y.ticks, callback: v => v + 's' } } },
    });

    // GFLOPS horizontal
    makeChart('chart-gflops', 'bar', names,
        [{ data: ordered().map(([, m]) => m.gflops), backgroundColor: PALETTE, borderRadius: 4, barThickness: 28 }],
        { ...chartDefaults, indexAxis: 'y', scales: { x: { ...chartDefaults.scales.y }, y: { ticks: { color: '#a1a1aa', font: { size: 11, weight: 500 } }, grid: { display: false } } } });
}

function makeChart(id, type, labels, datasets, options) {
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(document.getElementById(id), { type, data: { labels, datasets }, options });
}

// ── Model Details (Accordion) ──
function renderDetails() {
    const el = document.getElementById('model-details');
    el.innerHTML = ordered().map(([id, m], i) => `
        <details class="bg-surface-1 border border-zinc-800 rounded-xl group" id="detail-${id}">
            <summary class="px-5 py-3.5 cursor-pointer flex items-center justify-between hover:bg-surface-2/50 transition-colors rounded-xl list-none">
                <div class="flex items-center gap-3">
                    <span class="w-2.5 h-2.5 rounded-full" style="background:${PALETTE[i]}"></span>
                    <span class="text-sm font-semibold">${m.name}</span>
                    <span class="text-[11px] text-dim">${m.technique || ''}</span>
                </div>
                <svg class="w-4 h-4 text-dim transition-transform group-open:rotate-180" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>
            </summary>
            <div class="px-5 pb-5 pt-2 border-t border-zinc-800/60">
                <p class="text-sm text-subtle mb-4">${m.description || ''}</p>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-x-10 gap-y-1 text-xs">
                    ${detailRow('Parallelism', m.parallelism || '-')}
                    ${detailRow('Hardware', m.hardware || '-')}
                    ${detailRow('Train Samples', num(m.n_train))}
                    ${detailRow('Test Samples', num(m.n_test))}
                    ${detailRow('Features', m.n_features)}
                    ${detailRow('Classes', m.n_classes)}
                    ${detailRow('SVM Training', fmt(m.train_ms))}
                    ${detailRow('SVM Prediction', fmt(m.predict_ms))}
                    ${detailRow('DBSCAN', fmt(m.dbscan_ms))}
                    ${detailRow('Total Time', fmt(m.total_ms))}
                    ${detailRow('GFLOPS', m.gflops.toFixed(2))}
                    ${detailRow('FLOP Count', num(m.flops || 0))}
                    ${m.confident ? detailRow('Confident / Uncertain', `${num(m.confident)} / ${num(m.uncertain)}`) : ''}
                    ${m.dbscan_clusters ? detailRow('DBSCAN Clusters / Noise', `${m.dbscan_clusters} / ${m.dbscan_noise}`) : ''}
                </div>
            </div>
        </details>
    `).join('');
}

function detailRow(label, value) {
    return `<div class="flex justify-between py-1.5 border-b border-zinc-800/30"><span class="text-dim">${label}</span><span class="mono text-zinc-300">${value}</span></div>`;
}

// ── Helpers ──
function ordered() {
    return MODEL_ORDER.filter(id => models[id]).map(id => [id, models[id]]);
}

function fmt(ms) {
    if (!ms || ms === 0) return '—';
    if (ms < 1000) return ms.toFixed(0) + ' ms';
    if (ms < 60000) return (ms / 1000).toFixed(1) + ' s';
    return (ms / 60000).toFixed(1) + ' min';
}

function num(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString();
}
