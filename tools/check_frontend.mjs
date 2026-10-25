/**
 * 前端契约检查（CI 用）：页面里引用的 id / 静态资源是否真的存在。
 *
 * 为什么需要它：前端是三个 JS + 两个 HTML 拼起来的，改个 id 很容易漏掉
 * 另一侧的引用 —— 这类错误 pytest 看不见，浏览器里表现为「点了没反应」。
 * 这个脚本零依赖，node 直接跑，失败时返回非零退出码。
 */
import { readFileSync, existsSync } from "node:fs";

const PAGES = [
  { html: "static/index.html", js: "static/assets/workbench.js" },
  { html: "static/dashboard.html", js: "static/assets/dashboard.js" },
];

const problems = [];

for (const page of PAGES) {
  if (!existsSync(page.html)) { problems.push(`缺少页面：${page.html}`); continue; }
  const html = readFileSync(page.html, "utf8");
  const js = readFileSync(page.js, "utf8");

  // 1) 重复 id：getElementById 只会拿到第一个，另一个成为死元素
  const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]);
  const duplicated = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))];
  if (duplicated.length) problems.push(`${page.html} 存在重复 id：${duplicated.join(", ")}`);

  // 2) JS 里 $("#id") 引用的元素必须存在
  const refs = [...new Set([...js.matchAll(/\$\("#([A-Za-z0-9_-]+)"\)/g)].map((m) => m[1]))];
  const missingIds = refs.filter((id) => !ids.includes(id));
  if (missingIds.length) problems.push(`${page.js} 引用了不存在的 id：${missingIds.join(", ")}`);

  // 3) 页面引用的 /assets/* 必须存在
  for (const m of html.matchAll(/(?:href|src)="(\/assets\/[^"]+)"/g)) {
    if (!existsSync("static" + m[1])) problems.push(`${page.html} 引用了不存在的资源：${m[1]}`);
  }

  // 4) 每个页面都要引用共用的设计系统
  if (!html.includes("/assets/ui.css")) problems.push(`${page.html} 没有引用 /assets/ui.css`);
  if (!html.includes("/assets/ui.js")) problems.push(`${page.html} 没有引用 /assets/ui.js`);
}

if (problems.length) {
  console.error("前端契约检查失败：");
  for (const p of problems) console.error("  - " + p);
  process.exit(1);
}
console.log("前端契约检查通过：id / 静态资源 / 共用样式引用都完整");
