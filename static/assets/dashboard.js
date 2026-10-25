/* ==========================================================================
   LogWise · 观测台
   三块：4 个数字（健康吗）→ 一条曲线（什么时候出的问题）→ 一张表（哪一条）。
   数据全部来自 GET /stats；`?demo=1` 用内置样例渲染（截图 / 演示用，不连后端）。
   ========================================================================== */
(function () {
  "use strict";

  const { $, esc, toast, readError, fmtCost, fmtDuration, checkHealth } = window.LW;

  const API_REVISION = "5";
  const REFRESH_MS = 5000;
  const DEMO = new URLSearchParams(location.search).get("demo") === "1";

  const el = {
    content: $("#content"), empty: $("#empty"),
    kRequests: $("#kRequests"), kRequestsFoot: $("#kRequestsFoot"),
    kCache: $("#kCache"), kCacheBar: $("#kCacheBar"), kCacheFoot: $("#kCacheFoot"),
    kCost: $("#kCost"), kCostFoot: $("#kCostFoot"),
    kAgent: $("#kAgent"), kAgentFoot: $("#kAgentFoot"),
    p95: $("#p95Badge"), chart: $("#chart"), axis: $("#chartAxis"),
    recentBody: $("#recentBody"), uptime: $("#uptime"),
    auto: $("#autoRefresh"), btnReset: $("#btnReset"),
    rangeSel: $("#rangeSel"),
    modelChip: $("#modelChip"), demoChip: $("#demoChip"),
  };

  const state = { timer: null, loading: false, rows: [], paused: false, range: "1h" };

  /* 演示数据：结构必须和 /stats 的 metrics 一致 */
  const DEMO_STATS = {
    model: "deepseek-chat",
    metrics: {
      uptime_seconds: 3600,
      requests: { total: 128, errors: 3, llm_calls: 74, cache_hits: 41, cache_hit_rate: 0.63, billable_calls: 74 },
      latency_ms: { avg: 1100, p95: 1900, max: 6400, sample_size: 128 },
      tokens: { prompt: 213340, cached_prompt: 191000, completion: 31200 },
      cost: { total: 0.142, avg_per_billable_call: 0.0019 },
      agent: { requests: 18, tool_call_rate: 0.37, direct_answer_rate: 0.63, avg_tool_calls: 0.6, avg_rounds: 1.4, forced_answer_count: 2 },
      daily: { date: "2026-10-25", used_yuan: 0.012, requests: 128, llm_calls: 74, limit_yuan: 5, remaining_yuan: 4.988, exceeded: false },
      timeseries: [
        { minute: "2026-10-25T21:52", requests: 2, errors: 0, llm_calls: 2, cache_hits: 0, prompt_tokens: 2104, completion_tokens: 312, cost: 0.0013 },
        { minute: "2026-10-25T21:57", requests: 3, errors: 0, llm_calls: 2, cache_hits: 1, prompt_tokens: 3200, completion_tokens: 480, cost: 0.0018 },
        { minute: "2026-10-25T22:02", requests: 2, errors: 1, llm_calls: 2, cache_hits: 0, prompt_tokens: 1900, completion_tokens: 290, cost: 0.0012 },
        { minute: "2026-10-25T22:07", requests: 4, errors: 0, llm_calls: 3, cache_hits: 1, prompt_tokens: 5100, completion_tokens: 720, cost: 0.0031 },
        { minute: "2026-10-25T22:12", requests: 3, errors: 0, llm_calls: 3, cache_hits: 0, prompt_tokens: 4300, completion_tokens: 610, cost: 0.0026 },
        { minute: "2026-10-25T22:17", requests: 5, errors: 0, llm_calls: 3, cache_hits: 2, prompt_tokens: 6200, completion_tokens: 880, cost: 0.0034 },
        { minute: "2026-10-25T22:22", requests: 4, errors: 0, llm_calls: 4, cache_hits: 0, prompt_tokens: 5600, completion_tokens: 760, cost: 0.0038 },
        { minute: "2026-10-25T22:27", requests: 6, errors: 0, llm_calls: 3, cache_hits: 3, prompt_tokens: 7400, completion_tokens: 1020, cost: 0.0041 },
        { minute: "2026-10-25T22:32", requests: 5, errors: 1, llm_calls: 4, cache_hits: 1, prompt_tokens: 6800, completion_tokens: 940, cost: 0.0046 },
        { minute: "2026-10-25T22:37", requests: 7, errors: 0, llm_calls: 4, cache_hits: 3, prompt_tokens: 9100, completion_tokens: 1240, cost: 0.0052 },
        { minute: "2026-10-25T22:42", requests: 6, errors: 0, llm_calls: 3, cache_hits: 3, prompt_tokens: 7800, completion_tokens: 1090, cost: 0.0044 },
        { minute: "2026-10-25T22:47", requests: 8, errors: 1, llm_calls: 5, cache_hits: 3, prompt_tokens: 10400, completion_tokens: 1480, cost: 0.0061 },
        { minute: "2026-10-25T22:51", requests: 6, errors: 0, llm_calls: 4, cache_hits: 2, prompt_tokens: 8100, completion_tokens: 1150, cost: 0.0050 },
      ],
      // 与后端 /stats 一致：按时间正序（渲染时再倒过来显示最新在上）
      recent: [
        { time: "2026-10-25 22:48:58", endpoint: "/parse", duration_ms: 9, prompt_tokens: 0, cached_prompt_tokens: 0, completion_tokens: 0, cost: 0, cache_hit: false, error: "", request_id: "5e6f1a2b3c4d" },
        { time: "2026-10-25 22:49:30", endpoint: "/diagnose/followup", duration_ms: 2100, prompt_tokens: 2600, cached_prompt_tokens: 2200, completion_tokens: 380, cost: 0.0008, cache_hit: false, error: "", request_id: "9d4c3e77b120" },
        { time: "2026-10-25 22:49:55", endpoint: "/diagnose", duration_ms: 340, prompt_tokens: 0, cached_prompt_tokens: 0, completion_tokens: 0, cost: 0, cache_hit: false, error: "模型输出不符合格式（缺少字段 root_cause）", request_id: "7a2e5b91cc08" },
        { time: "2026-10-25 22:50:12", endpoint: "/context", duration_ms: 11, prompt_tokens: 0, cached_prompt_tokens: 0, completion_tokens: 0, cost: 0, cache_hit: false, error: "", request_id: "c3f0a81d55e2" },
        { time: "2026-10-25 22:50:41", endpoint: "/diagnose/stream", duration_ms: 1900, prompt_tokens: 2050, cached_prompt_tokens: 1800, completion_tokens: 300, cost: 0.0013, cache_hit: true, error: "", request_id: "1b7d90c2aa41" },
        { time: "2026-10-25 22:51:07", endpoint: "/diagnose/agent/stream", duration_ms: 4200, prompt_tokens: 2104, cached_prompt_tokens: 1890, completion_tokens: 312, cost: 0.0021, cache_hit: false, error: "", request_id: "8f2c1a9b3d4e" },
      ],
    },
  };

  /* ------------------------------------------------------------------ 启动 */
  init();

  function init() {
    checkHealth(API_REVISION);
    if (DEMO) {
      el.demoChip.hidden = false;
      el.auto.disabled = true;
      el.btnReset.disabled = true;
      el.rangeSel.disabled = true;   // 演示数据是固定的，切换范围没有意义
      render(DEMO_STATS);
      return;
    }
    bind();
    load();
  }

  function bind() {
    el.auto.addEventListener("change", schedule);
    el.btnReset.addEventListener("click", resetStats);
    el.recentBody.addEventListener("click", onRowClick);
    el.rangeSel.addEventListener("change", () => {
      state.range = el.rangeSel.value;
      load();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stopTimer();
      else schedule();
    });
    schedule();
  }

  function schedule() {
    stopTimer();
    if (el.auto.checked && !document.hidden) {
      state.timer = setInterval(load, REFRESH_MS);
    }
  }

  function stopTimer() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  /* ------------------------------------------------------------------ 取数 */
  async function load() {
    if (state.loading) return;
    // 展开某行的原始记录时暂停刷新：否则 5 秒后整表重建，正在看的那行会消失
    if (state.paused) return;
    state.loading = true;
    try {
      const resp = await fetch("/stats?range=" + encodeURIComponent(state.range));
      if (!resp.ok) throw new Error(await readError(resp));
      render(await resp.json());
      window.LW.setStatus(true, "运行中");
    } catch (e) {
      window.LW.setStatus(false, "未连接");
      if (!state.timer) toast("读取 /stats 失败：" + e.message, 3000);
    } finally {
      state.loading = false;
    }
  }

  async function resetStats() {
    if (!window.confirm("确定要清零统计与缓存吗？\n（清理后看到的数字就只包含之后的调用）")) return;
    try {
      const resp = await fetch("/stats/reset", { method: "POST" });
      if (!resp.ok) throw new Error(await readError(resp));
      toast("统计与缓存已清零");
      setPaused(false);   // 重置后必须真的刷新，不能被「展开详情暂停」挡住
      await load();
    } catch (e) {
      toast("重置失败：" + e.message, 3000);
    }
  }

  /** 展开详情时暂停轮询；收起后立刻补一次刷新 */
  function setPaused(on) {    state.paused = on;
    const hint = $("#recentHint");
    if (hint) {
      hint.textContent = on ? "已暂停刷新 · 收起详情后继续" : "点行展开原始记录";
    }
    if (!on) load();
  }

  /* ------------------------------------------------------------------ 渲染 */
  /** 时间范围 → 中文标签。故意写在函数内部：
   *  `init()` 在模块顶部就会执行，若把这张表写成模块级 const，
   *  render() 调用时它还在暂时性死区里（TDZ），会抛 ReferenceError。 */
  function rangeLabel(range) {
    if (range === "6h") return "最近 6 小时";
    if (range === "24h") return "最近 24 小时";
    return "最近 1 小时";
  }

  function updateChartTitle() {
    const title = $("#chartTitle");
    if (title) title.textContent = `成本与请求（${rangeLabel(state.range)}）`;
  }

  function render(stats) {
    if (stats.model) el.modelChip.textContent = "model: " + stats.model;
    const m = stats.metrics || {};
    const total = (m.requests && m.requests.total) || 0;

    if (!total) {
      el.content.hidden = true;
      el.empty.hidden = false;
      el.uptime.textContent = "";
      return;
    }
    el.empty.hidden = true;
    el.content.hidden = false;

    el.uptime.textContent = "已运行 " + fmtUptime(m.uptime_seconds);
    updateChartTitle();
    renderKpis(m);
    renderChart(m.timeseries || []);
    renderRecent(m.recent || []);
  }

  function fmtUptime(seconds) {
    if (!seconds) return "—";
    if (seconds < 3600) return Math.round(seconds / 60) + " 分钟";
    return (seconds / 3600).toFixed(1) + " 小时";
  }

  function renderKpis(m) {
    const req = m.requests || {};
    const lat = m.latency_ms || {};
    const cost = m.cost || {};
    const agent = m.agent || {};

    el.kRequests.innerHTML = `${req.total || 0} <small>次</small>`;
    el.kRequestsFoot.innerHTML =
      `错误 <b class="${(req.errors || 0) > 0 ? "warn" : ""}">${req.errors || 0}</b>` +
      ` · 真正调模型 ${req.llm_calls || 0} 次` +
      (lat.p95 ? ` · P95 ${(lat.p95 / 1000).toFixed(1)}s` : "");

    const rate = typeof req.cache_hit_rate === "number" ? req.cache_hit_rate : 0;
    el.kCache.innerHTML = `${Math.round(rate * 100)}<small>%</small>`;
    el.kCacheBar.style.width = Math.round(rate * 100) + "%";
    el.kCacheFoot.textContent = req.cache_hits
      ? `命中 ${req.cache_hits} 次 · 这些请求没花模型钱`
      : "还没有重复日志命中缓存";

    el.kCost.innerHTML = fmtCost(cost.total || 0);
    // 今日累计 vs 每日预算：这是 guard.ensure_budget 的实际生效数字
    const daily = m.daily || {};
    const foot = [
      req.billable_calls
        ? `计费 ${req.billable_calls} 次 · 均价 ${fmtCost(cost.avg_per_billable_call || 0)}`
        : "还没有产生费用",
    ];
    if (daily.limit_yuan > 0) {
      foot.push(`今日 ${fmtCost(daily.used_yuan || 0)} / ¥${daily.limit_yuan}`);
      if (daily.exceeded) foot.push('<b class="warn">已超预算，诊断被拦截</b>');
    } else if (daily.used_yuan) {
      foot.push(`今日 ${fmtCost(daily.used_yuan)}`);
    }
    el.kCostFoot.innerHTML = foot.join(" · ");

    if (agent.requests) {
      el.kAgent.innerHTML = `${Math.round((agent.tool_call_rate || 0) * 100)}<small>%</small>`;
      el.kAgentFoot.textContent =
        `${agent.requests} 次 · 直接回答 ${Math.round((agent.direct_answer_rate || 0) * 100)}%` +
        ` · 平均 ${agent.avg_rounds || 0} 轮` +
        (agent.forced_answer_count ? ` · 强制收尾 ${agent.forced_answer_count}` : "");
    } else {
      el.kAgent.innerHTML = `—`;
      el.kAgentFoot.textContent = "还没有 Agent 请求";
    }

    el.p95.textContent = lat.p95 ? `P95 ${(lat.p95 / 1000).toFixed(1)}s` : "P95 —";
  }

  /** 手写 SVG：一条面积（花费）+ 一条虚线（请求数），不引第三方图表库 */
  function renderChart(points) {
    const W = 1100, H = 230, TOP = 12, BOTTOM = 200;
    if (!points.length) {
      el.chart.innerHTML = `<text x="550" y="115" text-anchor="middle" fill="#5b6b80"
        font-size="13" font-family="Consolas, monospace">还没有时间序列数据</text>`;
      el.axis.innerHTML = "<span>—</span><span>—</span><span>—</span>";
      return;
    }

    // 24 小时 = 1440 个点，直接画既费 CPU 又糊成一团；抽稀到 ~240 个点足够看趋势
    if (points.length > 240) {
      const step = Math.ceil(points.length / 240);
      points = points.filter((_, i) => i % step === 0 || i === points.length - 1);
    }

    const n = points.length;
    const costs = points.map((p) => p.cost || 0);
    const reqs = points.map((p) => p.requests || 0);
    const maxCost = Math.max.apply(null, costs.concat([1e-6]));
    const maxReq = Math.max.apply(null, reqs.concat([1]));

    const x = (i) => (n === 1 ? 0 : (i / (n - 1)) * W);
    const yCost = (v) => BOTTOM - (v / maxCost) * (BOTTOM - TOP);
    const yReq = (v) => BOTTOM - (v / maxReq) * (BOTTOM - TOP) * 0.85;

    const costPts = points.map((p, i) => `${x(i).toFixed(1)},${yCost(p.cost || 0).toFixed(1)}`);
    const reqPts = points.map((p, i) => `${x(i).toFixed(1)},${yReq(p.requests || 0).toFixed(1)}`);
    const area = `M0,${BOTTOM} L${costPts.join(" L")} L${W},${BOTTOM} Z`;

    el.chart.innerHTML = `
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#22d3ee" stop-opacity=".34"/>
          <stop offset="1" stop-color="#22d3ee" stop-opacity="0"/>
        </linearGradient>
      </defs>
      <g stroke="#16202c" stroke-width="1">
        <line x1="0" y1="${TOP + 30}" x2="${W}" y2="${TOP + 30}"/>
        <line x1="0" y1="${(TOP + BOTTOM) / 2}" x2="${W}" y2="${(TOP + BOTTOM) / 2}"/>
        <line x1="0" y1="${BOTTOM - 40}" x2="${W}" y2="${BOTTOM - 40}"/>
      </g>
      <path d="${area}" fill="url(#areaGrad)"/>
      <path d="M${costPts.join(" L")}" stroke="#22d3ee" stroke-width="2"
            stroke-linejoin="round" fill="none"/>
      <path d="M${reqPts.join(" L")}" stroke="#a78bfa" stroke-width="1.4"
            stroke-dasharray="4 4" opacity=".75" fill="none"/>`;

    const label = (i) => String(points[i].minute || "").slice(-5);
    el.axis.innerHTML = [0, Math.floor((n - 1) / 2), n - 1]
      .map((i) => `<span>${esc(label(i))}</span>`).join("");
  }

  function renderRecent(rows) {
    state.rows = rows.slice().reverse();   // 最新的在最上面
    if (!state.rows.length) {
      el.recentBody.innerHTML = `<tr><td colspan="6" style="color:var(--aux)">还没有记录</td></tr>`;
      return;
    }
    el.recentBody.innerHTML = state.rows.map((r, i) => {
      const status = r.error
        ? `<span class="pill err" title="${esc(r.error)}">错误</span>`
        : (r.cache_hit ? `<span class="pill cache">缓存</span>` : `<span class="pill ok">ok</span>`);
      const tokens = (r.prompt_tokens || 0) + (r.completion_tokens || 0);
      return `<tr data-i="${i}" style="cursor:pointer">
        <td class="mono">${esc(String(r.time || "").slice(11))}</td>
        <td class="ep">${esc(r.endpoint || "")}</td>
        <td class="mono">${r.duration_ms ? fmtDuration(r.duration_ms) : "—"}</td>
        <td class="mono">${tokens || "—"}</td>
        <td class="mono">${fmtCost(r.cost || 0)}</td>
        <td>${status}</td>
      </tr>`;
    }).join("");
  }

  /** 点一行 → 展开原始记录（含 request_id，方便和诊断页对上号） */
  function onRowClick(e) {
    const tr = e.target.closest("tr[data-i]");
    if (!tr) return;
    const next = tr.nextElementSibling;
    if (next && next.classList.contains("raw-row")) {
      next.remove();
      setPaused(false);   // 收起后恢复轮询并补一次刷新
      return;
    }
    const row = state.rows[Number(tr.dataset.i)];
    if (!row) return;
    const pretty = JSON.stringify(row, null, 2);
    const detail = document.createElement("tr");
    detail.className = "raw-row";
    detail.innerHTML = `<td colspan="6" style="padding:0 20px 16px">
      <pre class="mono" style="margin:0;padding:12px;border-radius:9px;background:#0a1119;
        border:1px solid var(--line);color:#a9c4d8;font-size:11px;line-height:1.7;
        white-space:pre-wrap;word-break:break-all">${esc(pretty)}</pre></td>`;
    tr.after(detail);
    setPaused(true);
  }
})();
