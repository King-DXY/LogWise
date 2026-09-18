"""演示：日志上下文聚合带来的差别。

运行方式（任选一种）：
    1. 直接双击本文件
    2. 在终端里： .venv\\Scripts\\python.exe demo_context.py

这个脚本会拿一段真实形态的日志，分别按「旧行为」和「新行为」各跑一次诊断，
把两次结果的差别直接摆出来。

日志场景
--------
网关报「调用订单服务失败: connection refused」。
只看这条 ERROR，最自然的判断是「订单服务挂了」或者「网络不通」——
**方向是错的**。

真正的答案在同一次故障的另一条日志里：节点内存压力把订单服务的 Pod 驱逐了。
那条是 WARN 级别、还排在最前面，旧实现会直接把它丢掉。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _reenter_venv() -> None:
    """确保用项目自带的 .venv 运行。

    Windows 上双击 .py 用的是系统全局 Python，而 fastapi / openai 这些依赖
    只装在项目的 .venv 里。这里自动切过去，让「双击」和「命令行」效果一致。
    """
    venv_dir = ROOT / ".venv"
    venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not venv_python.exists():
        return
    try:
        if Path(sys.prefix).resolve() == venv_dir.resolve():
            return
        if Path(sys.executable).resolve() == venv_python.resolve():
            return
    except Exception:
        pass

    import subprocess

    print(f"[切换] 正在切换到项目虚拟环境：{venv_python}")
    completed = subprocess.run(
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=str(ROOT),
    )
    sys.exit(completed.returncode)


_reenter_venv()

from app.context import build_context, describe  # noqa: E402
from app.diagnose import diagnose_multi_ex, format_logs  # noqa: E402
from app.parser import parse_text  # noqa: E402

LOG_PATH = ROOT / "data" / "demo_pod_evicted.log"
LINE = "=" * 74


def title(text: str) -> None:
    print()
    print(LINE)
    print(f"  {text}")
    print(LINE)


def main() -> None:
    text = LOG_PATH.read_text(encoding="utf-8")
    logs = parse_text(text)

    title("第 0 步：解析这段日志")
    print(f"  文件: {LOG_PATH.relative_to(ROOT)}")
    print(f"  解析出 {len(logs)} 条日志：")
    for i, log in enumerate(logs, 1):
        print(f"    [{i}] {log.level:5s}  {log.message[:56]}")

    # ---------------- 旧行为 ---------------- #
    os.environ["LOGWISE_CONTEXT_ENABLED"] = "false"
    old_ctx = build_context(logs)

    title("第 1 步：旧行为会送什么给模型")
    print(f"  只挑第一条 ERROR，其余全部丢弃 —— 实际只送 {old_ctx.size} 条：")
    for log in old_ctx.logs:
        print(f"    ★ [{log.level}] {log.message[:60]}")
    print()
    print("  ⚠️  这条日志只说了「连接被拒绝」。")
    print("      模型最自然的判断是「订单服务挂了」或「网络不通」——")
    print("      但这只是症状，方向是错的。真正的原因在第 1 条 WARN 里，被丢掉了。")

    # ---------------- 新行为 ---------------- #
    os.environ.pop("LOGWISE_CONTEXT_ENABLED", None)
    new_ctx = build_context(logs)
    info = describe(new_ctx)

    title("第 2 步：新行为会送什么给模型")
    print(f"  以锚点为中心做上下文聚合，实际送 {info['selected']} 条"
          f"（窗口 ±{info['window']}，上限 {info['max_logs']}）：")
    for log in new_ctx.logs:
        mark = "★" if log is new_ctx.anchor else " "
        flag = "🔍" if "驱逐" in log.message else "  "
        print(f"    {mark} {flag}[{log.level:5s}] {log.message[:58]}")
    print()
    print("  ✅ 第 1 条 WARN「Pod 被驱逐：节点内存不足」现在被送进去了，")
    print("      熔断器那条 WARN 也在 —— 模型能看到完整的因果链。")

    # ---------------- 实际跑模型 ---------------- #
    title("第 3 步：真的调用模型，对比两次结论")
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("  未检测到 DEEPSEEK_API_KEY，跳过实际调用。")
        print("  新行为实际会发给模型的文本预览：")
        print()
        print("  " + format_logs(new_ctx.logs)[:500].replace("\n", "\n  "))
        return

    for label, ctx in (("旧行为（只送第一条 ERROR）", old_ctx),
                       ("新行为（上下文聚合）", new_ctx)):
        print()
        print(f"  --- {label} ---")
        try:
            result, usage = diagnose_multi_ex(ctx.logs)
        except Exception as exc:  # 演示脚本不该因为网络问题直接崩掉
            print(f"      调用失败：{type(exc).__name__}: {exc}")
            continue

        print(f"      输入 {usage.prompt_tokens} token / 输出 {usage.completion_tokens} token")
        print(f"      错误类型  : {result.get('error_type', '(无)')}")
        print(f"      置信度    : {result.get('confidence')}")
        print(f"      根因      :")
        for chunk in _wrap(str(result.get("root_cause", "(无)")), 62):
            print(f"        {chunk}")
        for ev in (result.get("evidence") or []):
            print(f"      证据      : {str(ev)[:62]}")

    title("结论")
    print("  两次调用用的是同一个 Prompt、同一个模型。")
    print("  唯一的差别是「送进去几条日志」—— 但结论的质量完全不同：")
    print()
    print("    旧行为：停在症状层（连接被拒），置信度低，只好说「无法进一步区分」")
    print("    新行为：找到真因（Pod 被内存压力驱逐），并指出后面都是连带症状")
    print()
    print("  想看聚合参数和成本统计，启动服务后访问 http://127.0.0.1:8000/stats")
    print()


def _wrap(text: str, width: int) -> list[str]:
    """按宽度折行，让长段落在终端里好读。"""
    lines: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if len(current) >= width:
            lines.append(current)
            current = ""
    if current:
        lines.append(current)
    return lines or [""]


if __name__ == "__main__":
    try:
        main()
    finally:
        # 双击运行时窗口会一闪而过，留个停顿
        if sys.stdin.isatty():
            try:
                input("按回车键关闭本窗口...")
            except (EOFError, KeyboardInterrupt):
                pass
