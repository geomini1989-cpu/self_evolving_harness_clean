import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "memory"

st.set_page_config(layout="wide", page_title="自进化智能体评测与优化平台")


THEME_COLORS = {
    "accepted": "#16875d",
    "rolled_back": "#c2410c",
    "buffered": "#9a6a00",
    "neutral": "#475569",
}


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.35rem; padding-bottom: 2rem; }
    h1, h2, h3 { letter-spacing: 0; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .hero-band {
        border: 1px solid #cbd8ea;
        border-radius: 8px;
        padding: 24px 24px 18px 24px;
        background:
            linear-gradient(90deg, rgba(255,255,255,0.95) 0%, rgba(241,247,255,0.96) 58%, rgba(246,248,241,0.96) 100%);
        margin-bottom: 16px;
        overflow: visible;
    }
    .hero-title {
        font-size: 30px;
        line-height: 1.45;
        font-weight: 760;
        color: #0f172a;
        margin: 0 0 6px 0;
        padding-top: 2px;
        overflow: visible;
    }
    .hero-subtitle {
        color: #334155;
        font-size: 15px;
        max-width: 1080px;
        line-height: 1.7;
    }
    .hero-tags {
        margin-top: 12px;
    }
    .hero-tag {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 10px;
        margin-right: 8px;
        margin-bottom: 4px;
        background: #ffffff;
        border: 1px solid #d7dee9;
        color: #334155;
        font-size: 12px;
        font-weight: 600;
    }
    .command-strip {
        margin-top: 10px;
        padding: 8px 10px;
        border-radius: 6px;
        background: #0f172a;
        color: #e2e8f0;
        font-family: Consolas, "Courier New", monospace;
        font-size: 12px;
        line-height: 1.6;
        display: block;
        width: fit-content;
        max-width: 100%;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .insight-box {
        border: 1px solid #d7dee9;
        border-radius: 8px;
        padding: 12px 14px;
        background: #fbfdff;
        color: #334155;
        margin: 8px 0 12px 0;
        line-height: 1.65;
    }
    .section-note {
        color: #64748b;
        font-size: 13px;
        margin-top: -6px;
        margin-bottom: 10px;
    }
    .status-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 2px 8px;
        font-size: 12px;
        border: 1px solid #cbd5e1;
        color: #334155;
        background: #f8fafc;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def path(name):
    return MEMORY_DIR / name


def coerce_numeric_column(series):
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() == 0 and series.notna().sum() > 0:
        return series
    return converted


def plot_chart(fig):
    st.plotly_chart(fig, width="stretch")


def show_table(df, **kwargs):
    st.dataframe(df, width="stretch", **kwargs)


def numeric_long_frame(df, index_col, value_cols, name_col="metric", value_col="value"):
    numeric_df = df[[index_col] + value_cols].copy()
    for col in value_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    long_df = numeric_df.melt(id_vars=index_col, value_vars=value_cols, var_name=name_col, value_name=value_col)
    return long_df.dropna(subset=[value_col])


@st.cache_data(ttl=3)
def read_csv(name):
    file_path = path(name)
    if not file_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(file_path, dtype=str, on_bad_lines="skip")
    if "ts" in df.columns:
        df = df[df["ts"] != "ts"]
    for col in df.columns:
        if col not in {"epoch", "model_type", "model"}:
            df[col] = coerce_numeric_column(df[col])
    return df


@st.cache_data(ttl=3)
def read_jsonl(name, limit=5000):
    file_path = path(name)
    if not file_path.exists():
        return []
    records = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records[-limit:]


@st.cache_data(ttl=3)
def read_json(name, default=None):
    file_path = path(name)
    if not file_path.exists():
        return {} if default is None else default
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {} if default is None else default


@st.cache_data(ttl=3)
def read_text(name):
    file_path = path(name)
    if not file_path.exists():
        return ""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_skill_blocks(skill_text):
    pattern = re.compile(r"##\s*\[(.*?)\]\s*(.*?)\n(.*?)(?=\n##|\Z)", re.DOTALL)
    rows = []
    for category, title, body in pattern.findall(skill_text or ""):
        block = f"## [{category}] {title}\n{body}".strip()
        rows.append(
            {
                "category": category.strip(),
                "title": title.strip(),
                "chars": len(block),
                "approx_tokens": max(1, len(block) // 4),
                "body": body.strip(),
            }
        )
    return pd.DataFrame(rows)


def metric_value(df, column, default=0):
    if df.empty or column not in df.columns:
        return default
    value = df.iloc[-1].get(column, default)
    try:
        return float(value)
    except Exception:
        return default


def event_summary(version_records):
    events = [record.get("event") for record in version_records]
    accepted = sum(1 for event in events if str(event).startswith("accepted"))
    rolled_back = sum(1 for event in events if "rolled_back" in str(event))
    candidates = sum(1 for event in events if event == "candidate_written")
    return accepted, rolled_back, candidates


def artifact_summary(skill_text, prompt_policy, examples, tips, rejected):
    skill_count = len(parse_skill_blocks(skill_text))
    example_count = len(examples) if isinstance(examples, list) else 0
    promoted_tips = sum(1 for tip in tips if tip.get("status") == "promoted")
    buffered_tips = sum(1 for tip in tips if tip.get("status") == "buffered")
    return {
        "Skill": skill_count,
        "Few-shot": example_count,
        "Prompt Policy": 1 if prompt_policy.strip() else 0,
        "Promoted Tips": promoted_tips,
        "Buffered Tips": buffered_tips,
        "Rejected": len(rejected),
    }


def render_status_pills(summary):
    labels = []
    for name, value in summary.items():
        labels.append(f'<span class="status-pill">{name}: {value}</span>')
    st.markdown("".join(labels), unsafe_allow_html=True)


def f1_run_summary(metrics_df):
    if metrics_df.empty or "f1_score" not in metrics_df.columns:
        return 0.0, 0.0, 0.0, 0.0
    values = pd.to_numeric(metrics_df["f1_score"], errors="coerce").dropna()
    if values.empty:
        return 0.0, 0.0, 0.0, 0.0
    first = float(values.iloc[0])
    current = float(values.iloc[-1])
    peak = float(values.max())
    return first, current, peak, current - first


def render_global_header():
    st.markdown(
        """
        <div class="hero-band">
            <div class="hero-title">自进化智能体评测与优化平台</div>
            <div class="hero-subtitle">
            冻结基座模型参数，通过批量执行、自动评估、根因反思、低风险进化和回归门控，验证 Agent 在低 Token 成本下的持续优化能力。
            </div>
            <div class="hero-tags">
                <span class="hero-tag">SkillOS 技能记忆</span>
                <span class="hero-tag">SkillOpt 策略优化</span>
                <span class="hero-tag">Self-Harness 闭环进化</span>
                <span class="hero-tag">Few-shot 轻量晋升</span>
                <span class="hero-tag">防退化回滚</span>
            </div>
            <div class="command-strip">推荐正式演示：python main_loop.py --mode llm-evolve --sample-size 300 --epochs 6 --batch-size 8 --max-workers 2 --reset-state</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_overview(metrics_df, token_df, versions, tips, rejected, skill_text, prompt_policy, examples, transfer_report):
    latest_f1 = metric_value(metrics_df, "f1_score")
    latest_calls = metric_value(metrics_df, "llm_calls")
    latest_tokens = metric_value(metrics_df, "total_tokens")
    latest_elapsed = metric_value(metrics_df, "elapsed_ms")
    cache_hits = metric_value(metrics_df, "cache_hits")
    accepted, rolled_back, candidates = event_summary(versions)
    transfer_score = float(transfer_report.get("avg_score") or 0.0) if transfer_report else 0.0
    first_f1, current_f1, peak_f1, delta_f1 = f1_run_summary(metrics_df)
    examples_count = len(examples) if isinstance(examples, list) else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("当前 F1", f"{current_f1:.3f}")
    c2.metric("峰值 F1", f"{peak_f1:.3f}")
    c3.metric("较首轮", f"{delta_f1:+.3f}")
    c4.metric("Few-shot 资产", f"{examples_count}")
    c5.metric("回滚保护", f"{rolled_back}")
    c6.metric("Token", f"{int(latest_tokens)}")

    artifact_counts = artifact_summary(skill_text, prompt_policy, examples, tips, rejected)
    render_status_pills(
        {
            "LLM 调用": int(latest_calls),
            "缓存命中": int(cache_hits),
            "耗时 ms": int(latest_elapsed),
            "迁移分": f"{transfer_score:.2f}",
            "候选补丁": candidates,
            "已接受": accepted,
            "已回滚": rolled_back,
            **artifact_counts,
        }
    )
    st.markdown(
        f"""
        <div class="insight-box">
        当前实验首轮 F1={first_f1:.3f}，峰值 F1={peak_f1:.3f}，当前 F1={current_f1:.3f}。
        若曲线末段小幅回落，应结合回滚次数和 Few-shot 资产一起看：系统正在优先沉淀低风险样例，并拒绝会破坏回归集的强规则补丁。
        Skill=0 不代表没有进化，当前策略优先使用 Few-shot 轻量晋升，Skill 只在重复稳定错误出现后作为强规则资产写入。
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1])
    with left:
        if not metrics_df.empty:
            chart_df = metrics_df.copy()
            chart_df["run_index"] = range(1, len(chart_df) + 1)
            fig = px.line(
                chart_df,
                x="run_index",
                y="f1_score",
                markers=True,
                title="F1 演化曲线（关注峰值、当前值和回滚保护）",
                color_discrete_sequence=["#2563eb"],
            )
            fig.update_layout(height=330, margin=dict(l=12, r=12, t=48, b=12))
            fig.update_yaxes(range=[0.8, 1.0])
            plot_chart(fig)
        else:
            st.info("尚未生成 metrics.csv")
    with right:
        if versions:
            rows = pd.DataFrame({"event": [record.get("event") for record in versions]})
            counts = rows["event"].value_counts().reset_index()
            counts.columns = ["event", "count"]
            fig = px.bar(
                counts.head(12),
                x="count",
                y="event",
                orientation="h",
                title="进化事件分布",
                color="event",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(height=330, showlegend=False, margin=dict(l=12, r=12, t=48, b=12))
            plot_chart(fig)
        else:
            st.info("尚未生成 skill_versions.jsonl")

    if not token_df.empty and {"model_type", "total_tokens"}.issubset(token_df.columns):
        grouped = token_df.groupby("model_type", dropna=False)["total_tokens"].sum().reset_index()
        fig = px.bar(grouped, x="model_type", y="total_tokens", title="Token 消耗按模型类型统计", color="model_type")
        fig.update_layout(height=280, showlegend=False, margin=dict(l=12, r=12, t=48, b=12))
        plot_chart(fig)


def render_evolution(version_records):
    st.subheader("进化与防退化")
    st.markdown('<div class="section-note">候选补丁写入后必须经过 sample、replay、full 回归门控，失败会恢复写入前状态。</div>', unsafe_allow_html=True)
    if not version_records:
        st.info("尚未生成进化记录")
        return

    rows = []
    for record in version_records:
        metrics = record.get("metrics", {}) or {}
        rows.append(
            {
                "ts": record.get("ts"),
                "event": record.get("event"),
                "action": record.get("evolution_action"),
                "target_category": record.get("target_category"),
                "root_cause": record.get("root_cause_type"),
                "confidence": record.get("confidence"),
                "tip_id": record.get("tip_id"),
                "baseline_f1": metrics.get("baseline_f1"),
                "sample_f1": metrics.get("sample_f1"),
                "replay_f1": metrics.get("replay_f1"),
                "full_f1": metrics.get("full_f1"),
            }
        )
    df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    accepted, rolled_back, candidates = event_summary(version_records)
    c1.metric("候选补丁", candidates)
    c2.metric("接受补丁", accepted)
    c3.metric("回滚补丁", rolled_back)

    f1_cols = [col for col in ["baseline_f1", "sample_f1", "replay_f1", "full_f1"] if col in df.columns]
    f1_df = df[f1_cols].dropna(how="all")
    if not f1_df.empty:
        f1_df = f1_df.reset_index(names="step")
        long_f1 = numeric_long_frame(f1_df, "step", f1_cols)
        if not long_f1.empty:
            fig = px.line(long_f1, x="step", y="value", color="metric", markers=True, title="回归门控结果")
            fig.update_layout(height=330, margin=dict(l=12, r=12, t=48, b=12))
            plot_chart(fig)

    show_table(df.tail(80), hide_index=True)


def render_memory(skill_text, prompt_policy, examples, tips, rejected):
    st.subheader("记忆资产")
    skill_df = parse_skill_blocks(skill_text)
    example_count = len(examples) if isinstance(examples, list) else 0
    prompt_count = 1 if prompt_policy.strip() else 0
    promoted = sum(1 for tip in tips if tip.get("status") == "promoted")
    buffered = sum(1 for tip in tips if tip.get("status") == "buffered")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("强规则 Skill", len(skill_df))
    c2.metric("Few-shot 学习样例", example_count)
    c3.metric("Prompt Policy", prompt_count)
    c4.metric("Promoted Tip", promoted)
    c5.metric("Rejected", len(rejected))

    st.markdown(
        """
        <div class="insight-box">
        当前采用 Few-shot-first 策略：先把 Bad Case 固化为低风险样例，只有重复稳定的错误模式才晋升为强规则 Skill。
        因此 Few-shot 数量是主要自进化产物，Skill=0 表示系统尚未接受需要全局生效的强规则。
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.write("强规则 SkillRepo（可选晋升）")
        if skill_df.empty:
            st.info("暂无已接受强规则 Skill。当前已有 Few-shot 学习资产，系统仍在采用更低风险的样例记忆。")
        else:
            counts = skill_df.groupby("category").size().reset_index(name="count")
            plot_chart(px.bar(counts, x="category", y="count", title="Skill 按类别分布"))
            show_table(skill_df[["category", "title", "chars", "approx_tokens"]], hide_index=True)
    with right:
        st.write("Few-shot 学习样例（主要长期记忆）")
        if not isinstance(examples, list) or not examples:
            st.info("暂无 Few-shot 示例")
        else:
            example_rows = [
                {
                    "input": str(item.get("input", ""))[:80],
                    "core_intent": (item.get("output") or {}).get("core_intent"),
                    "source": item.get("source"),
                }
                for item in examples[-80:]
            ]
            show_table(pd.DataFrame(example_rows), hide_index=True)

    with st.expander("查看 Prompt Policy"):
        st.markdown(prompt_policy or "暂无 Prompt Policy")
    with st.expander("查看 SKILL.md"):
        st.markdown(skill_text or "暂无 SKILL.md")

    if tips:
        tip_df = pd.DataFrame(
            [
                {
                    "tip_id": tip.get("tip_id"),
                    "status": tip.get("status"),
                    "count": tip.get("count"),
                    "action": tip.get("evolution_action"),
                    "root_cause": tip.get("root_cause_type"),
                    "confidence": tip.get("confidence"),
                    "rule": str(tip.get("proposed_rule") or "")[:100],
                }
                for tip in tips
            ]
        )
        status_counts = tip_df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        fig = px.bar(status_counts, x="status", y="count", title=f"双层记忆状态：buffered={buffered}, promoted={promoted}")
        fig.update_layout(height=300, margin=dict(l=12, r=12, t=48, b=12))
        plot_chart(fig)
        show_table(tip_df.tail(100), hide_index=True)


def render_transfer_test(report, transfer_traces):
    st.subheader("迁移测试")
    if not report:
        st.info("尚未生成 transfer_report.json")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Suite 数", int(report.get("suite_count") or 0))
    c2.metric("Case 数", int(report.get("case_count") or 0))
    c3.metric("平均迁移分", f"{float(report.get('avg_score') or 0):.2f}")

    suites = pd.DataFrame(report.get("suites") or [])
    if not suites.empty:
        fig = px.bar(suites, x="domain", y="avg_score", color="modality", title="跨领域迁移评分")
        fig.update_layout(height=330, margin=dict(l=12, r=12, t=48, b=12))
        plot_chart(fig)
        show_table(suites, hide_index=True)

    if transfer_traces:
        rows = []
        for record in transfer_traces:
            state = record.get("state", {}) or {}
            action = record.get("action", {}) or {}
            feedback = record.get("feedback", {}) or {}
            rows.append(
                {
                    "domain": state.get("domain"),
                    "modality": state.get("modality"),
                    "action": action.get("name"),
                    "score": feedback.get("score"),
                    "exact_match": feedback.get("exact_match"),
                    "errors": len(feedback.get("errors") or []),
                }
            )
        show_table(pd.DataFrame(rows), hide_index=True)


def render_saf_traces(trace_records):
    st.subheader("State-Action-Feedback 轨迹")
    if not trace_records:
        st.info("尚未生成 SAF 轨迹")
        return

    rows = []
    for record in trace_records:
        state = record.get("state", {}) or {}
        action = record.get("action", {}) or {}
        feedback = record.get("feedback", {}) or {}
        rows.append(
            {
                "domain": state.get("domain"),
                "modality": state.get("modality"),
                "action": action.get("name"),
                "score": feedback.get("score"),
                "exact_match": feedback.get("exact_match"),
                "errors": len(feedback.get("errors") or []),
                "input": str(state.get("raw_input", ""))[:100],
            }
        )
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trace 数", len(df))
    c2.metric("平均反馈分", f"{pd.to_numeric(df['score'], errors='coerce').mean():.2f}")
    c3.metric("Exact Match", f"{pd.to_numeric(df['exact_match'], errors='coerce').mean():.2%}")

    grouped = df.groupby(["domain", "modality"], dropna=False).size().reset_index(name="count")
    fig = px.bar(grouped, x="domain", y="count", color="modality", title="领域与模态分布")
    fig.update_layout(height=320, margin=dict(l=12, r=12, t=48, b=12))
    plot_chart(fig)
    show_table(df.tail(200), hide_index=True)


def render_latest_patch(patch):
    st.subheader("最新归因")
    if not patch:
        st.info("尚未生成 latest_patch.json")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("根因类型", str(patch.get("root_cause_type") or patch.get("error_category") or "-"))
    c2.metric("进化动作", str(patch.get("evolution_action") or "-"))
    c3.metric("置信度", f"{float(patch.get('confidence') or 0):.2f}")
    st.caption(f"目标类别: {patch.get('target_category') or '-'}")
    st.write("根因分析")
    st.write(patch.get("root_cause_analysis") or "-")
    st.write("建议规则")
    st.code(patch.get("proposed_rule") or "-", language="markdown")
    with st.expander("完整 patch JSON"):
        st.json(patch)


metrics = read_csv("metrics.csv")
tokens = read_csv("token_usage.csv")
versions = read_jsonl("skill_versions.jsonl")
traces = read_jsonl("saf_traces.jsonl", limit=3000)
tips = read_jsonl("tips.jsonl", limit=3000)
rejected = read_jsonl("rejected_skills.jsonl", limit=3000)
transfer_report = read_json("transfer_report.json")
transfer_traces = read_jsonl("transfer_traces.jsonl", limit=3000)
skill_text = read_text("SKILL.md")
prompt_policy = read_text("PROMPT_POLICY.md")
examples = read_json("examples.json", default=[])
latest_patch = read_json("latest_patch.json")

render_global_header()

tab_demo, tab_evolution, tab_memory, tab_transfer, tab_trace, tab_patch = st.tabs(
    ["总览", "自进化闭环", "记忆资产", "迁移验证", "执行轨迹", "归因详情"]
)

with tab_demo:
    render_demo_overview(metrics, tokens, versions, tips, rejected, skill_text, prompt_policy, examples, transfer_report)

with tab_evolution:
    render_evolution(versions)

with tab_memory:
    render_memory(skill_text, prompt_policy, examples, tips, rejected)

with tab_transfer:
    render_transfer_test(transfer_report, transfer_traces)

with tab_trace:
    render_saf_traces(traces)

with tab_patch:
    render_latest_patch(latest_patch)

if st.button("刷新数据"):
    st.cache_data.clear()
    st.rerun()
