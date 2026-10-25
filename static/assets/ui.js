/* ==========================================================================
   LogWise · 前端共用工具
   只放两个页面都会用的东西：选择器、转义、提示条、错误处理、格式化。
   ========================================================================== */
(function (global) {
  "use strict";

  /** 单元素 / 多元素选择器 */
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  /**
   * HTML 转义。所有来自模型 / 用户的文本都必须过这一道再拼进 innerHTML，
   * 否则日志里带一段 <img onerror=...> 就是一次 XSS。
   */
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  /** 轻提示，取代 alert() */
  let toastTimer = null;
  function toast(message, ms) {
    const el = $("#toast");
    if (!el) return;
    el.textContent = message;
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), ms || 2000);
  }

  /** 从失败响应里取一句人话 */
  async function readError(resp) {
    let raw = "";
    try {
      raw = await resp.text();
    } catch (e) {
      return "HTTP " + resp.status;
    }
    try {
      const body = JSON.parse(raw);
      return String(body.detail || body.error || raw);
    } catch (e) {
      return raw || "HTTP " + resp.status;
    }
  }

  function httpError(status, detail) {
    const err = new Error(detail || "HTTP " + status);
    err.status = status;
    return err;
  }

  const isAbort = (e) => e && e.name === "AbortError";

  /** 一次 POST（JSON body） */
  async function postJSON(url, body, options) {
    const opts = Object.assign({ method: "POST", headers: { "Content-Type": "application/json" } }, options || {});
    opts.body = JSON.stringify(body);
    return fetch(url, opts);
  }

  /** 把 `/stats` 的三档单价读出来算一次成本（流式接口偶尔不带 usage） */
  function fmtCost(yuan) {
    if (yuan == null || Number.isNaN(yuan)) return "—";
    if (yuan === 0) return "¥0";
    if (yuan < 0.0001) return "¥" + yuan.toFixed(6);
    return "¥" + yuan.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  }

  function fmtDuration(ms) {
    if (ms < 1000) return Math.round(ms) + " ms";
    if (ms < 10000) return (ms / 1000).toFixed(1) + "s";
    return Math.round(ms / 1000) + "s";
  }

  /** 顶栏的连接状态灯 */
  function setStatus(online, text) {
    const chip = $("#statusChip");
    const label = $("#statusText");
    if (!chip) return;
    chip.classList.toggle("live", !!online);
    chip.classList.toggle("offline", !online);
    if (label) label.textContent = text || (online ? "在线" : "未连接");
  }

  /** 服务端版本探测：网页是新版、后端还是旧代码时，必须让用户知道 */
  async function checkHealth(expectedApi) {
    try {
      const resp = await fetch("/health");
      if (!resp.ok) throw new Error("health " + resp.status);
      const info = await resp.json();
      setStatus(true, "在线");
      const ver = $("#verChip");
      if (ver) ver.textContent = "v0.4 · api " + info.api;
      if (expectedApi && info.api !== expectedApi) {
        const banner = $("#staleBanner");
        if (banner) banner.classList.add("show");
      }
      return info;
    } catch (e) {
      setStatus(false, "未连接");
      return null;
    }
  }

  /** 逐行读 SSE；回调拿到 JSON.parse 后的 data，[DONE] 结束 */
  async function readSSE(resp, onEvent) {
    if (!resp.body) throw new Error("当前浏览器不支持流式响应");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;   // ": ping" 心跳在这被忽略
        const data = line.slice(6);
        if (data === "[DONE]") return true;
        let parsed;
        try {
          parsed = JSON.parse(data);
        } catch (e) {
          throw new Error("收到无法解析的服务器事件");
        }
        onEvent(parsed);
      }
    }
    return false;   // 没等到 [DONE] 就断开 = 连接中断
  }

  global.LW = {
    $, $$, esc, toast, readError, httpError, isAbort, postJSON,
    fmtCost, fmtDuration, setStatus, checkHealth, readSSE,
  };
})(window);
