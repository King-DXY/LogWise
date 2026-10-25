/* ==========================================================================
   LogWise · 工作台
   --------------------------------------------------------------------------
   流程：贴日志 → /context（免费，先看送了哪些）→ /diagnose/agent/stream 或
        /diagnose/stream → 边流边把字段填进卡片 → 完成后用完整对象再渲染一次。

   为什么流式也要「按字段填卡」而不是直接把 JSON 打出来：
   用户等的是结论，不是花括号。这里用一个增量解析器，字段一闭合就先显示。
   ========================================================================== */
(function () {
  "use strict";

  const { $, esc, toast, readError, isAbort, postJSON, fmtCost, fmtDuration, checkHealth, readSSE } = window.LW;

  const API_REVISION = "5";   // 与后端 /health 的 api 字段对齐
  // ?demo=1：用内置样例走一遍完整渲染（不连后端）。截图、演示、以及"页面上线前
  // 先确认结果态长什么样"都用它 —— 否则没有 API Key 就永远看不到结果卡。
  const DEMO = new URLSearchParams(location.search).get("demo") === "1";

  const el = {
    composer: $("#composer"), result: $("#result"),
    logInput: $("#logInput"), btnRun: $("#btnRun"), btnClear: $("#btnClear"),
    samples: $("#samples"),
    logSummary: $("#logSummary"), agentTag: $("#agentTag"),
    btnEdit: $("#btnEdit"), btnRetry: $("#btnRetry"), btnCancel: $("#btnCancel"), btnCopy: $("#btnCopy"),
    dKicker: $("#dKicker"), dTitle: $("#dTitle"), dRoot: $("#dRoot"), dError: $("#dError"),
    dNext: $("#dNext"), dNextText: $("#dNextText"), dAfter: $("#dAfter"),
    ringFg: $("#ringFg"), ringNum: $("#ringNum"),
    mElapsed: $("#mElapsed"), mCost: $("#mCost"), mTools: $("#mTools"), mReqId: $("#mReqId"),
    foldEvidence: $("#foldEvidence"), eviList: $("#eviList"), eviCount: $("#eviCount"),
    foldCtx: $("#foldCtx"), ctxList: $("#ctxList"), ctxCount: $("#ctxCount"),
    foldTrace: $("#foldTrace"), traceList: $("#traceList"), traceCount: $("#traceCount"),
    chat: $("#chat"), askInput: $("#askInput"), askSend: $("#askSend"),
    modelChip: $("#modelChip"),
    resumeBar: $("#resumeBar"), btnBack: $("#btnBack"),
    resumeAt: $("#resumeAt"), resumeHint: $("#resumeHint"),
  };

  const RING_LEN = 2 * Math.PI * 39;   // r=39 的周长，用于置信度环

  const state = {
    running: false,
    controller: null,
    timer: null,
    startedAt: 0,
    text: "",
    sessionId: null,
    ctx: null,
    settled: false,
    restoring: false,
    toolCalls: 0,
    steps: [],
    pending: null,   // 追问请求的 AbortController
    saved: null,     // 本地保存的上一次诊断（用于「返回上次诊断」）
  };

  /* ------------------------------------------------------------ 本地持久化 */
  /* 为什么必须有：工作台和观测台是两个独立页面，点导航是整页跳转。
     一次诊断是花过钱的，session_id 也只活 30 分钟 —— 去观测台看一眼再回来，
     结果和会话就全丢了，等于白花钱。所以把「上一次结果 + 草稿」存在本地：
       · 回到工作台 → 直接恢复上次的诊断（含追问会话）
       · 中途刷新 → 粘贴的日志也不会丢
     只保留 30 分钟（与会话 TTL 对齐），过期就丢掉，免得展示一个问不了的旧会话。 */
  const STORE_KEY = "logwise.workbench.result.v1";
  const DRAFT_KEY = "logwise.workbench.draft.v1";
  const KEEP_MS = 30 * 60 * 1000;

  function saveResult(payload, elapsedMs) {
    const record = {
      at: Date.now(),
      elapsedMs: elapsedMs || 0,
      text: state.text,
      ctx: state.ctx,
      requestId: el.mReqId.textContent,
      payload: payload,
    };
    state.saved = record;
    if (DEMO) return;
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(record));
    } catch (e) { /* 隐私模式下 localStorage 可能不可用，忽略即可 */ }
  }

  function readResult() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
      if (!saved || !saved.payload || !saved.payload.diagnosis) return null;
      if (Date.now() - saved.at > KEEP_MS) { localStorage.removeItem(STORE_KEY); return null; }
      return saved;
    } catch (e) {
      return null;
    }
  }

  let draftTimer = null;
  function saveDraft(text) {
    clearTimeout(draftTimer);
    draftTimer = setTimeout(() => {
      try { localStorage.setItem(DRAFT_KEY, text); } catch (e) {}
    }, 300);
  }

  function clearSaved() {
    try {
      localStorage.removeItem(STORE_KEY);
      localStorage.removeItem(DRAFT_KEY);
    } catch (e) {}
  }

  /** 把上一次的诊断原样摆回来（不重新请求，也就不会再花钱） */
  function restore(saved) {
    state.text = saved.text || "";
    state.saved = saved;
    state.toolCalls = 0;   // 恢复的是历史快照，计数必须从 0 开始重新累加
    state.ctx = saved.ctx || null;
    state.startedAt = performance.now() - (saved.elapsedMs || 0);
    // 只有在输入区还空着时才回填日志 —— 否则会把用户刚改了一半的草稿冲掉
    if (!el.logInput.value.trim()) el.logInput.value = state.text;

    showResult();
    resetCard();
    if (state.ctx) { renderLogbar(state.ctx, state.text); renderContext(state.ctx); }
    if (saved.requestId) el.mReqId.textContent = saved.requestId;

    const rounds = (saved.payload.trace && saved.payload.trace.rounds) || [];
    rounds.forEach((r) => {
      if (r.decision === "tool_call") addTraceRound(r.round, r.tool_calls || []);
    });

    state.restoring = true;
    finalize(saved.payload);
    state.restoring = false;

    const at = new Date(saved.at);
    const hh = String(at.getHours()).padStart(2, "0");
    const mm = String(at.getMinutes()).padStart(2, "0");
    setKicker("● 上次诊断 · " + hh + ":" + mm, "done");
    el.btnCancel.hidden = true;
    el.btnRetry.hidden = false;
    el.resumeBar.hidden = true;
  }

  /* ------------------------------------------------------------------ 启动 */
  init();

  function init() {
    bind();
    // 草稿先填回去：刷新 / 跳转回来后，粘进去的日志还在
    try {
      const draft = localStorage.getItem(DRAFT_KEY);
      if (draft) el.logInput.value = draft;
    } catch (e) { /* localStorage 不可用就算了 */ }

    checkHealth(API_REVISION).then((info) => {
      if (!info) return;
      fetch("/stats").then((r) => r.json()).then((s) => {
        if (el.modelChip) el.modelChip.textContent = "model: " + s.model;
      }).catch(() => {});
    });

    if (DEMO) {
      runDemo();
      return;
    }
    // 从观测台切回来时，直接把上一次的诊断摆回来（不重新请求、不再花钱）
    const saved = readResult();
    if (saved) { state.saved = saved; restore(saved); }
  }

  function bind() {
    el.btnRun.addEventListener("click", run);
    el.btnClear.addEventListener("click", clearAll);
    el.btnEdit.addEventListener("click", showComposer);
    el.btnBack.addEventListener("click", () => { if (state.saved) restore(state.saved); });
    // 整条日志栏都可点：用户第一反应是「点上面那行」而不是找按钮
    $("#logbar").addEventListener("click", (e) => {
      if (e.target.closest("button")) return;   // 按钮自己处理，别重复触发
      showComposer();
    });
    el.btnRetry.addEventListener("click", () => { if (state.running) return; showComposer(); run(); });
    el.btnCancel.addEventListener("click", cancelRun);
    el.btnCopy.addEventListener("click", copyDiagnosis);

    el.samples.addEventListener("click", (e) => {
      const btn = e.target.closest(".sample");
      if (btn) loadSample(btn.dataset.file);
    });

    el.askSend.addEventListener("click", askFollowup);
    el.askInput.addEventListener("keydown", (e) => { if (e.key === "Enter") askFollowup(); });

    el.eviList.addEventListener("click", (e) => {
      const li = e.target.closest("li");
      if (li) locateEvidence(li.dataset.text || "");
    });

    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        // 正在跑的时候不要重复触发（更不能顺手取消），只提示一下
        if (state.running) toast("正在诊断中，稍等…");
        else run();
      }
      if (e.key === "Escape" && state.running) cancelRun();
    });

    el.logInput.addEventListener("input", () => {
      if (el.logInput.value.trim()) el.btnRun.disabled = false;
      saveDraft(el.logInput.value);
    });

    // 诊断进行中离开页面 = 中断请求、已花的 token 拿不回结果 —— 先问一句
    window.addEventListener("beforeunload", (e) => {
      if (state.running) { e.preventDefault(); e.returnValue = ""; }
    });
  }

  /* ------------------------------------------------------------ 示例与清空 */
  async function loadSample(file) {
    if (!file) return;
    try {
      const resp = await fetch("/samples/" + file);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      el.logInput.value = await resp.text();
      toast("已填入示例日志");
      el.logInput.focus();
    } catch (e) {
      toast("示例加载失败：" + e.message);
    }
  }

  function clearAll() {
    cancelRun();
    el.logInput.value = "";
    clearSaved();
    state.saved = null;
    el.resumeBar.hidden = true;
    showComposer();
    el.logInput.focus();
  }

  /* ---------------------------------------------------------------- 主流程 */
  async function run() {
    if (state.running) { cancelRun(); return; }
    // 演示模式完全离线：任何「开始诊断 / 重新诊断」都只是重播内置样例，
    // 绝不能悄悄发出真实请求 —— 否则没配 Key 会报错，配了 Key 会真花钱。
    if (DEMO) {
      runDemo();
      toast("演示模式：重播内置样例，不会调用模型");
      return;
    }
    const text = el.logInput.value.trim();
    if (!text) { toast("请先粘贴日志"); el.logInput.focus(); return; }

    state.running = true;
    state.settled = false;
    state.text = text;
    saveDraft(text);
    state.ctx = null;
    state.toolCalls = 0;
    state.steps = [];
    state.sessionId = null;
    state.startedAt = performance.now();
    state.controller = new AbortController();

    showResult();
    resetCard();
    startTimer();

    try {
      await loadContext(text);
      // Agent 模式：模型自己决定要不要查工具；普通模式：代码写死检索策略
      if ($("#agentMode").checked) await runAgent(text);
      else await runPlain(text);
    } catch (e) {
      if (!isAbort(e)) renderError(e);
    } finally {
      state.running = false;
      state.controller = null;
      stopTimer();
      el.btnCancel.hidden = true;
      el.btnRetry.disabled = false;
      el.btnRetry.hidden = false;
      el.btnCopy.hidden = state.settled ? false : true;
    }
  }

  function cancelRun() {
    if (!state.running) return;
    state.controller && state.controller.abort();
    state.running = false;
    stopTimer();
    el.btnCancel.hidden = true;
    el.btnRetry.hidden = false;
    el.btnRetry.disabled = false;
    el.btnCopy.hidden = true;
    setKicker("● 已取消", "");
    if (!state.settled) el.dTitle.textContent = "诊断已取消";
    toast("已取消");
  }

  /* ------------------------------------------------------------ 上下文预览 */
  async function loadContext(text) {
    el.logSummary.textContent = "正在解析日志…";
    try {
      const resp = await postJSON("/context", { text }, { signal: state.controller.signal });
      if (!resp.ok) throw new Error(await readError(resp));
      const ctx = await resp.json();
      state.ctx = ctx;
      renderLogbar(ctx, text);
      renderContext(ctx);
    } catch (e) {
      if (isAbort(e)) throw e;
      // 预览失败不阻断诊断 —— 它只是"顺手展示"，不是主链路
      el.logSummary.textContent = "解析预览失败 · 仍可继续诊断";
      el.ctxCount.textContent = "不可用";
    }
  }

  function renderLogbar(ctx, text) {
    const lines = text.split("\n").filter((l) => l.trim()).length;
    if (ctx && ctx.total_parsed) {
      el.logSummary.innerHTML =
        `<b>${lines} 行日志</b><span class="sep">·</span>解析出 ${ctx.total_parsed} 条` +
        `<span class="sep">·</span>送模型 <b>${ctx.selected}</b> 条` +
        (ctx.truncated ? `<span class="sep">·</span>已按优先级裁剪` : "");
    } else {
      el.logSummary.innerHTML = `<b>${lines} 行日志</b><span class="sep">·</span>未解析出结构化日志`;
    }
  }

  function renderContext(ctx) {
    if (!ctx || !Array.isArray(ctx.logs)) { el.ctxCount.textContent = "不可用"; return; }
    el.ctxCount.textContent = `解析 ${ctx.total_parsed} 条 / 送模型 ${ctx.selected} 条`;
    el.ctxList.innerHTML = ctx.logs.map((log) => {
      const cls = log.selected ? (log.is_anchor ? "sel anchor" : "sel") : "skip";
      const tag = log.is_anchor
        ? '<span class="tag a">锚点</span>'
        : (log.selected ? '<span class="tag s">送模型</span>' : '<span class="tag x">丢弃</span>');
      return `<li class="logrow ${cls}" data-idx="${log.index}">` +
        `<span class="ln">${log.index}</span>` +
        `<span class="lv lv-${esc(log.level || "INFO")}">${esc(log.level || "")}</span>` +
        `<span class="msg">${esc(log.message || "")}</span>${tag}</li>`;
    }).join("");
  }

  /* -------------------------------------------------------------- Agent 模式 */
  async function runAgent(text) {
    el.agentTag.textContent = "Agent 模式";
    setKicker("● Agent 正在决策", "busy");
    const resp = await postJSON("/diagnose/agent/stream", { text }, { signal: state.controller.signal });
    if (!resp.ok) throw new Error(await readError(resp));
    el.mReqId.textContent = resp.headers.get("X-Request-ID") || "—";

    let buffer = "";
    let done = null;
    let streamError = "";

    const completed = await readSSE(resp, (ev) => {
      if (ev.type === "tool_call") {
        addTraceRound(ev.round, ev.tool_calls || []);
      } else if (ev.type === "delta") {
        buffer += ev.text || "";
        applyPartial(buffer);
      } else if (ev.type === "retry") {
        addTraceNote("输出格式不合格，已自动纠错重试：" + (ev.problem || ""));
      } else if (ev.type === "error") {
        streamError = ev.message || "Agent 诊断失败";
      } else if (ev.type === "done") {
        done = ev;
      }
    });

    if (!completed && !streamError) throw new Error("连接中断，未收到完整诊断结果");
    if (streamError) throw new Error(streamError);
    if (!done) throw new Error("没有收到诊断结果");
    finalize(done);
  }

  /* -------------------------------------------------------------- 普通模式 */
  async function runPlain(text) {
    el.agentTag.textContent = "普通模式";
    el.dKicker.textContent = "● 正在生成";
    const resp = await postJSON("/diagnose/stream", { text }, { signal: state.controller.signal });
    if (!resp.ok) throw new Error(await readError(resp));
    el.mReqId.textContent = resp.headers.get("X-Request-ID") || "—";
    el.traceCount.textContent = "普通模式未调用工具";

    let buffer = "";
    let sessionId = null;
    let streamError = "";

    const completed = await readSSE(resp, (ev) => {
      if (typeof ev === "string") {
        buffer += ev;
        applyPartial(buffer);
      } else if (ev.session_id) {
        sessionId = ev.session_id;
      } else if (ev.error) {
        streamError = ev.error;
      }
    });

    if (!completed && !streamError) throw new Error("连接中断，未收到完整诊断结果");
    if (streamError) throw new Error(streamError);
    let diagnosis;
    try {
      diagnosis = JSON.parse(buffer);
    } catch (e) {
      throw new Error("模型输出不完整或不是合法 JSON，请重试");
    }
    finalize({ diagnosis, session_id: sessionId, usage: null, trace: null, cached: false });
  }

  /* ------------------------------------------------- 增量解析：边流边填字段 */
  function readString(buffer, key) {
    const marker = '"' + key + '"';
    const at = buffer.indexOf(marker);
    if (at < 0) return null;
    let p = buffer.indexOf(":", at + marker.length);
    if (p < 0) return null;
    p++;
    while (p < buffer.length && /\s/.test(buffer[p])) p++;
    if (buffer[p] !== '"') return null;
    p++;
    let out = "";
    for (; p < buffer.length; p++) {
      const ch = buffer[p];
      if (ch === "\\") {
        const next = buffer[p + 1];
        if (next == null) break;
        out += ({ n: "\n", t: "\t", r: "\r", '"': '"', "\\": "\\", "/": "/" }[next] || next);
        p++;
        continue;
      }
      if (ch === '"') return { value: out, complete: true };
      out += ch;
    }
    return { value: out, complete: false };
  }

  function readArray(buffer, key) {
    const marker = '"' + key + '"';
    const at = buffer.indexOf(marker);
    if (at < 0) return null;
    const open = buffer.indexOf("[", at + marker.length);
    if (open < 0) return null;
    const slice = buffer.slice(open + 1);
    const items = [];
    let i = 0;
    while (i < slice.length) {
      const q = slice.indexOf('"', i);
      if (q < 0) break;
      let p = q + 1, out = "", closed = false;
      for (; p < slice.length; p++) {
        const ch = slice[p];
        if (ch === "\\") { out += slice[++p] || ""; continue; }
        if (ch === '"') { closed = true; break; }
        out += ch;
      }
      items.push(out);
      if (!closed) break;
      i = p + 1;
    }
    return items;
  }

  function readNumber(buffer, key) {
    const m = new RegExp('"' + key + '"\\s*:\\s*(-?[0-9.]+)').exec(buffer);
    return m ? Number(m[1]) : null;
  }

  /** 把已经流出来的部分字段填进卡片（最终仍以完整 JSON 为准） */
  function applyPartial(raw) {
    if (state.settled) return;
    const start = raw.indexOf("{");
    if (start < 0) return;
    const buffer = raw.slice(start);

    const type = readString(buffer, "error_type");
    if (type && type.value) {
      el.dTitle.textContent = type.value;
      setKicker("● 正在生成", "busy");
    }

    const root = readString(buffer, "root_cause");
    if (root) {
      el.dRoot.innerHTML = esc(root.value) + '<span class="caret"></span>';
    }

    const conf = readNumber(buffer, "confidence");
    if (conf != null && conf >= 0 && conf <= 1) drawRing(conf);

    const steps = readArray(buffer, "solution_steps");
    if (steps && steps.length) renderSteps(steps);

    const evidence = readArray(buffer, "evidence");
    if (evidence && evidence.length) renderEvidence(evidence);
  }

  /* ------------------------------------------------------------ 完成渲染 */
  function finalize(payload) {
    state.settled = true;
    const d = payload.diagnosis || {};
    const usage = payload.usage || null;
    const trace = payload.trace || null;

    el.dTitle.textContent = d.error_type || "未能识别错误类型";
    setKicker(payload.cached ? "● 命中缓存（本次未计费）" : "● 诊断完成",
      payload.cached ? "cached" : "done");
    el.dError.innerHTML = "";

    const impact = d.impact ? ` <span class="impact">影响：${esc(d.impact)}</span>` : "";
    el.dRoot.innerHTML = esc(d.root_cause || "（模型未给出根因）") + impact;

    if (typeof d.confidence === "number") drawRing(d.confidence);
    renderSteps(d.solution_steps || []);
    renderEvidence(d.evidence || []);

    if (trace) {
      state.toolCalls = trace.tool_calls_total || 0;
      el.traceCount.textContent = state.toolCalls + " 次工具调用" + (trace.forced_answer ? " · 已强制收尾" : "");
      if (trace.forced_answer) {
        el.traceList.insertAdjacentHTML("beforeend",
          '<li><span class="name warn-note">达到轮次上限</span><span class="res">已强制模型收尾（防成本失控）</span></li>');
      }
    }

    const elapsedMs = performance.now() - state.startedAt;
    el.mElapsed.textContent = fmtDuration(elapsedMs);
    // 命中缓存时不要把「首次生成」的成本当成这次的成本显示 —— 那样和
    // 「本次未计费」的标签自相矛盾（用户实测截图里就是 ¥0.0016 配这句话）
    if (usage && payload.cached) {
      el.mCost.textContent = "¥0（缓存）";
      el.mCost.title = "首次生成时花费 " + fmtCost(usage.cost_yuan);
    } else {
      el.mCost.textContent = usage ? fmtCost(usage.cost_yuan) : "—";
      el.mCost.title = "";
    }
    el.mTools.textContent = trace ? state.toolCalls + " 次工具" : "未调工具";
    el.btnCopy.hidden = false;
    // 恢复时不能再存一遍：否则「上次诊断」的时间戳会被不断刷新，30 分钟永远走不完
    if (!state.restoring) saveResult(payload, elapsedMs);

    if (payload.session_id) {
      state.sessionId = payload.session_id;
      el.askInput.disabled = false;
      el.askSend.disabled = false;
      el.askInput.placeholder = "就这次诊断追问，例如：第三步具体怎么操作？";
    }
  }

  /** 状态胶囊：进行中=青 / 完成=绿 / 命中缓存=紫 / 失败=粉（默认） */
  function setKicker(text, kind) {
    el.dKicker.textContent = text;
    el.dKicker.className = "kicker" + (kind ? " " + kind : "");
  }

  function drawRing(conf) {    const pct = Math.round(Math.max(0, Math.min(1, conf)) * 100);
    el.ringNum.textContent = pct + "%";
    el.ringFg.style.strokeDashoffset = String(RING_LEN * (1 - conf));
    el.ringNum.style.color = conf >= 0.7 ? "var(--text)" : (conf >= 0.5 ? "var(--warn)" : "var(--err)");
  }

  function renderSteps(steps) {
    state.steps = steps.filter((s) => String(s || "").trim());
    if (!state.steps.length) { el.dNext.hidden = true; el.dAfter.innerHTML = ""; return; }
    el.dNext.hidden = false;
    el.dNextText.innerHTML = esc(state.steps[0]);
    el.dAfter.innerHTML = state.steps.slice(1)
      .map((s) => `<li>${esc(s)}</li>`).join("");
  }

  function renderEvidence(items) {
    const list = items.filter((s) => String(s || "").trim());
    el.eviCount.textContent = list.length ? `${list.length} 条 · 点击定位到日志` : "0 条";
    el.eviList.innerHTML = list.map((text) =>
      `<li data-text="${esc(text)}"><code>${esc(text)}</code></li>`).join("");
    el.foldEvidence.hidden = list.length === 0;
  }

  /** 点证据 → 展开上下文并高亮对应日志行 */
  function locateEvidence(text) {
    if (!text || !state.ctx || !Array.isArray(state.ctx.logs)) return;
    const needle = text.slice(0, 24);
    const hit = state.ctx.logs.find((l) => (l.message || "").includes(needle));
    if (!hit) { toast("这条证据来自堆栈行，未出现在聚合视图里"); return; }
    el.foldCtx.open = true;
    const row = el.ctxList.querySelector(`.logrow[data-idx="${hit.index}"]`);
    if (!row) return;
    Array.from(el.ctxList.querySelectorAll(".flash")).forEach((x) => x.classList.remove("flash"));
    row.classList.add("flash");
    row.scrollIntoView({ block: "nearest", behavior: "smooth" });
    setTimeout(() => row.classList.remove("flash"), 1800);
  }

  /* ------------------------------------------------------------ Agent 轨迹 */
  function addTraceRound(round, calls) {
    state.toolCalls += calls.length;
    el.traceCount.textContent = state.toolCalls + " 次工具调用";
    calls.forEach((call) => {
      const args = prettyArgs(call.arguments);
      const res = String(call.result_excerpt || "").replace(/\s+/g, " ").slice(0, 90);
      el.traceList.insertAdjacentHTML("beforeend",
        `<li><span class="name">${esc(call.name || "tool")}</span>` +
        `<span class="res">${esc(args)}${res ? " → " + esc(res) : ""}</span></li>`);
    });
    if (!state.running) el.traceCount.textContent = state.toolCalls + " 次工具调用";
  }

  function addTraceNote(text) {
    el.traceList.insertAdjacentHTML("beforeend",
      `<li><span class="name warn-note">重试</span><span class="res">${esc(text)}</span></li>`);
  }

  function prettyArgs(raw) {
    if (!raw) return "";
    try {
      const obj = JSON.parse(raw);
      return Object.entries(obj).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join("，");
    } catch (e) {
      return String(raw).slice(0, 60);
    }
  }

  /* ---------------------------------------------------------------- 追问 */
  async function askFollowup() {
    const question = el.askInput.value.trim();
    if (!question || !state.sessionId || state.pending) return;
    state.pending = new AbortController();
    el.askSend.disabled = true;

    el.chat.insertAdjacentHTML("beforeend", `<div class="bubble-q">${esc(question)}</div>`);
    const answerId = "a" + Date.now();
    el.chat.insertAdjacentHTML("beforeend",
      `<div class="bubble-a pending" id="${answerId}">思考中…</div>`);
    el.askInput.value = "";

    try {
      const resp = await postJSON("/diagnose/followup",
        { session_id: state.sessionId, question }, { signal: state.pending.signal });
      const node = document.getElementById(answerId);
      if (!resp.ok) {
        const detail = await readError(resp);
        node.textContent = "追问失败：" + detail;
        node.classList.remove("pending");
        if (resp.status === 404) {
          state.sessionId = null;
          el.askInput.disabled = true;
          el.askSend.disabled = true;
        }
        return;
      }
      const body = await resp.json();
      node.classList.remove("pending");
      node.textContent = body.answer || "（空回答）";
      const u = body.usage || {};
      node.insertAdjacentHTML("beforeend",
        `<div class="chat-cost">输入 ${u.prompt_tokens || 0} tok（缓存命中 ${u.cached_tokens || 0}）＋ 输出 ${u.completion_tokens || 0} tok</div>`);
    } catch (e) {
      if (!isAbort(e)) {
        const node = document.getElementById(answerId);
        if (node) { node.textContent = "追问失败：" + e.message; node.classList.remove("pending"); }
      }
    } finally {
      state.pending = null;
      el.askSend.disabled = !state.sessionId;
      el.askInput.disabled = !state.sessionId;
      if (state.sessionId) el.askInput.focus();
    }
  }

  /* ---------------------------------------------------------------- 视图切换 */
  function showComposer() {
    cancelRun();
    el.result.hidden = true;
    el.composer.hidden = false;
    syncResumeBar();
    el.logInput.focus();
  }

  /** 输入页 → 结果页的入口：有历史结果时才出现 */
  function syncResumeBar() {
    if (!state.saved) { el.resumeBar.hidden = true; return; }
    const at = new Date(state.saved.at);
    const hh = String(at.getHours()).padStart(2, "0");
    const mm = String(at.getMinutes()).padStart(2, "0");
    el.resumeAt.textContent = " · " + hh + ":" + mm;
    el.resumeHint.textContent = "上次的结果还在，不会重新计费";
    el.resumeBar.hidden = false;
  }

  function showResult() {
    el.composer.hidden = true;
    el.result.hidden = false;
    el.btnCancel.hidden = false;
    el.btnRetry.hidden = true;
    el.btnCopy.hidden = true;
    el.chat.innerHTML = "";
    el.askInput.disabled = true;
    el.askSend.disabled = true;
    el.askInput.value = "";
  }

  function resetCard() {
    setKicker("● 诊断中", "busy");
    el.dTitle.textContent = "正在分析日志…";
    el.dRoot.innerHTML = '<span class="caret"></span>';
    el.dError.innerHTML = "";
    el.dNext.hidden = true;
    el.dAfter.innerHTML = "";
    el.ringNum.textContent = "—";
    el.ringFg.style.strokeDashoffset = String(RING_LEN);
    el.mElapsed.textContent = "0.0s";
    el.mCost.textContent = "—";
    el.mCost.title = "";
    el.mTools.textContent = "—";
    el.mReqId.textContent = "—";
    el.eviList.innerHTML = "";
    el.eviCount.textContent = "0 条";
    el.foldEvidence.hidden = true;
    el.ctxList.innerHTML = "";
    el.ctxCount.textContent = "—";
    el.traceList.innerHTML = "";
    el.traceCount.textContent = "—";
    el.agentTag.textContent = ($("#agentMode").checked ? "Agent 模式" : "普通模式");
  }

  function startTimer() {
    stopTimer();
    const tick = () => {
      el.mElapsed.textContent = fmtDuration(performance.now() - state.startedAt) + "…";
    };
    tick();
    state.timer = setInterval(tick, 100);
  }

  function stopTimer() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  /* ---------------------------------------------------------------- 错误 */
  function renderError(err) {
    const status = err.status || 0;
    const message = err.message || String(err);
    setKicker("● 诊断失败", "");
    el.dTitle.textContent = "这次没有拿到结果";
    el.dRoot.innerHTML = "";
    el.dNext.hidden = true;
    el.dAfter.innerHTML = "";
    el.mCost.textContent = "—";
    el.btnCopy.hidden = true;

    let title = "诊断失败";
    let steps = [];
    if (message.includes("DEEPSEEK_API_KEY")) {
      title = "还没配置 API Key";
      steps = [
        "把项目里的 <code>.env.example</code> 复制一份、改名为 <code>.env</code>",
        "打开 <code>.env</code>，把 <code>DEEPSEEK_API_KEY=</code> 后面换成你的 Key",
        "关掉黑色命令行窗口，重新双击 <code>app\\run.py</code>",
      ];
    } else if (status === 429) {
      // 429 现在有两种来源，提示完全不同：限流（太快）和每日预算（钱花完了）
      if (message.includes("预算")) {
        title = "今日模型预算已用完";
        steps = [
          "明天会按自然日自动恢复",
          "急用可以在 <code>.env</code> 里调大 <code>LOGWISE_DAILY_BUDGET_YUAN</code>（0 = 不限制）",
          "改完要重启 <code>app\\run.py</code> 才生效",
        ];
      } else {
        title = "请求太频繁，被限流了";
        steps = ["稍等约 1 分钟再试（默认 20 次 / 60 秒）", "连续调试时可以先用「重新诊断」等一等"];
      }
    } else if (status === 413) {
      title = "日志太长了";
      steps = ["先截取故障时间段（建议只贴锚点前后几百行）", "需要放宽上限可在 .env 里改 <code>LOGWISE_MAX_LOG_CHARS</code>"];
    } else if (status === 502) {
      title = "模型这次没能给出诊断";
      steps = ["再点一次「重新诊断」（偶发失败重试通常能成功）", "连续失败时打开 <a href=\"/stats\" style=\"color:#22d3ee\">/stats</a> 查看详细错误"];
    } else if (message.includes("Failed to fetch") || message.includes("NetworkError")) {
      title = "连不上本地服务";
      steps = ["确认 <code>app\\run.py</code> 那个窗口还开着", "刷新页面再试"];
    }

    const html = steps.length
      ? `<h3>${esc(title)}</h3><ol>${steps.map((s) => `<li>${s}</li>`).join("")}</ol>` +
        `<div class="raw">技术细节：${esc(message.slice(0, 300))}</div>`
      : `<h3>诊断失败</h3><div class="raw">${esc(message)}</div>`;
    el.dError.innerHTML = `<div class="errbox">${html}</div>`;
  }

  /* ------------------------------------------------------------ 演示模式 */
  function runDemo() {
    const text = [
      "2026-09-17 22:48:10,900 INFO  [monitor] com.example.monitor.HealthChecker - health check ok 200",
      "2026-09-17 22:48:11,102 INFO  [scheduler-1] com.example.report.ReportJob - 开始生成日报 reportDate=2026-09-17",
      "2026-09-17 22:48:12,517 WARN  [scheduler-1] com.example.report.ReportJob - 写入文件重试 1/3: /data/reports/2026-09-17.csv",
      "2026-09-17 22:48:13,884 WARN  [scheduler-1] com.example.report.ReportJob - 写入文件重试 2/3: /data/reports/2026-09-17.csv",
      "2026-09-17 22:48:15,220 WARN  [scheduler-1] com.example.report.ReportJob - 写入文件重试 3/3: /data/reports/2026-09-17.csv",
      "2026-09-17 22:48:15,231 ERROR [scheduler-1] com.example.report.ReportJob - 日报生成失败 reportDate=2026-09-17",
      "java.io.IOException: No space left on device",
      "\tat com.example.report.CsvWriter.writeRow(CsvWriter.java:93)",
      "2026-09-17 22:49:02,663 ERROR [http-nio-8080-exec-7] com.example.report.ReportController - 报表下载接口异常 reportId=RPT20260917",
      "org.springframework.web.util.NestedServletException: Request processing failed",
      "\tat com.example.report.ReportService.load(ReportService.java:78)",
      "Caused by: java.io.UncheckedIOException: java.io.IOException: No space left on device",
    ].join("\n");

    const logs = [
      { index: 1, level: "INFO", message: "开始生成日报 reportDate=2026-09-17", selected: true, is_anchor: false },
      { index: 2, level: "WARN", message: "写入文件重试 1/3: /data/reports/2026-09-17.csv", selected: true, is_anchor: false },
      { index: 5, level: "ERROR", message: "日报生成失败 reportDate=2026-09-17", selected: true, is_anchor: true },
      { index: 6, level: "INFO", message: "health check ok 200", selected: false, is_anchor: false },
      { index: 7, level: "INFO", message: "metrics pushed: 42 points", selected: false, is_anchor: false },
      { index: 8, level: "ERROR", message: "报表下载接口异常 reportId=RPT20260917", selected: true, is_anchor: false },
    ];

    const diagnosis = {
      error_type: "磁盘空间不足导致文件写入失败",
      root_cause: "写入 /data/reports/2026-09-17.csv 时抛 No space left on device（ENOSPC），是内核在 write(2) 阶段直接拒绝，不是权限或句柄问题；5 分钟后报表下载接口因同一原因 500，可排除单个文件损坏。",
      evidence: [
        "java.io.IOException: No space left on device",
        "at com.example.report.CsvWriter.writeRow(CsvWriter.java:93)",
        "Caused by: java.io.IOException: No space left on device",
      ],
      impact: "日报导出当日无产出、报表下载全量 500、下游统计缺一天。",
      solution_steps: [
        "清理历史报表与过期日志，或给 /data 扩容",
        "检查日志轮转策略是否失效（本次疑似历史导出未清理）",
        "清理后重跑 ReportJob，补生成当日日报",
      ],
      confidence: 0.87,
    };

    el.logInput.value = text;
    showResult();
    resetCard();
    state.text = text;
    state.startedAt = performance.now() - 4200;
    state.ctx = { total_parsed: 12, selected: 7, logs: logs };
    el.mReqId.textContent = "8f2c1a9b3d4e";
    el.agentTag.textContent = "Agent 模式";
    renderLogbar({ total_parsed: 12, selected: 7, truncated: false }, text);
    renderContext(state.ctx);

    addTraceRound(1, [{ name: "get_log_overview", arguments: "{}", result_excerpt: "12 条 · ERROR 2 / WARN 3 / INFO 7" }]);
    addTraceRound(2, [{ name: "search_fault_cases", arguments: '{"query":"No space left on device","k":3}', result_excerpt: "命中 3 条案例 · disk_full / io_error" }]);

    finalize({
      diagnosis: diagnosis,
      usage: { prompt_tokens: 2104, cached_tokens: 1890, completion_tokens: 312, cost_yuan: 0.0013 },
      trace: { rounds: [], tool_calls_total: 2, forced_answer: false },
      session_id: "demo-session",
      cached: false,
    });
    el.btnCancel.hidden = true;
    el.btnRetry.hidden = false;
    const demoChip = $("#demoChip");
    if (demoChip) demoChip.hidden = false;
  }

  /* ---------------------------------------------------------------- 复制 */
  /** 复制到剪贴板：优先用现代 API，失败再退回 textarea + execCommand。
   *
   * 为什么要兜底：navigator.clipboard 只在「安全上下文」（https 或
   * localhost / 127.0.0.1）里存在。用局域网 IP 打开、或浏览器策略受限时，
   * 直接调用会抛 TypeError —— 表现就是「点了没反应」。 */
  async function copyText(text) {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch (e) { /* 落到下面的兜底 */ }

    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.cssText = "position:fixed;top:-1000px;left:0;opacity:0";
      document.body.appendChild(ta);
      ta.select();
      ta.setSelectionRange(0, text.length);
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) {
      return false;
    }
  }

  /** 拼成 Markdown：贴到工单、聊天、笔记本里都能直接用 */
  function buildMarkdown() {
    return [
      "## LogWise 诊断",
      "",
      `**错误类型**：${el.dTitle.textContent}`,
      `**置信度**：${el.ringNum.textContent}`,
      "",
      "### 根因",
      el.dRoot.textContent.trim(),
      "",
      "### 排查步骤",
      ...state.steps.map((s, i) => `${i + 1}. ${s}`),
      "",
      "---",
      `生成于 ${new Date().toLocaleString("zh-CN")}`,
    ].join("\n");
  }

  async function copyDiagnosis() {
    const btn = el.btnCopy;
    const label = btn.textContent;
    const ok = await copyText(buildMarkdown());
    // 反馈就放在按钮上 —— 之前只弹屏幕最底部的 toast，离按钮大半个屏幕，看着像没反应
    btn.textContent = ok ? "已复制 ✓" : "复制失败";
    toast(ok ? "已复制为 Markdown，可直接粘贴" : "复制失败：请手动选中内容复制", 2000);
    setTimeout(() => { btn.textContent = label; }, 1800);
  }
})();
