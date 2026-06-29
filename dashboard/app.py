import json
import os
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MEMORY_DIR = ROOT / "memory"


st.set_page_config(layout="wide", page_title="Self-Evolving Harness Dashboard")


def path(name):
    return MEMORY_DIR / name


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
            df[col] = pd.to_numeric(df[col], errors="ignore")
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
def read_json(name):
    file_path = path(name)
    if not file_path.exists():
        return {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


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


def render_overview(metrics_df, token_df):
    st.subheader("运行总览")
    latest_f1 = metric_value(metrics_df, "f1_score")
    latest_calls = metric_value(metrics_df, "llm_calls")
    latest_tokens = metric_value(metrics_df, "total_tokens")
    latest_elapsed = metric_value(metrics_df, "elapsed_ms")
    cache_hits = metric_value(metrics_df, "cache_hits")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新 F1", f"{latest_f1:.2f}")
    c2.metric("LLM 调用", f"{int(latest_calls)}")
    c3.metric("Token", f"{int(latest_tokens)}")
    c4.metric("缓存命中", f"{int(cache_hits)}")
    c5.metric("耗时 ms", f"{int(latest_elapsed)}")

    if not metrics_df.empty:
        chart_df = metrics_df.copy()
        chart_df["run_index"] = range(1, len(chart_df) + 1)
        st.plotly_chart(
            px.line(chart_df, x="run_index", y="f1_score", markers=True, title="F1 迭代曲线"),
            use_container_width=True,
        )
        cost_cols = [col for col in ["llm_calls", "cache_hits", "total_tokens", "elapsed_ms"] if col in chart_df.columns]
        if cost_cols:
            st.plotly_chart(
                px.line(chart_df, x="run_index", y=cost_cols, markers=True, title="调用、缓存、Token 与延迟"),
                use_container_width=True,
            )
    else:
        st.info("尚未生成 metrics.csv。先运行 main_loop.py 后刷新。")

    if not token_df.empty and {"model_type", "total_tokens"}.issubset(token_df.columns):
        grouped = token_df.groupby("model_type", dropna=False)["total_tokens"].sum().reset_index()
        st.plotly_chart(px.bar(grouped, x="model_type", y="total_tokens", title="按模型类型统计 Token"), use_container_width=True)


def render_evolution(version_records):
    st.subheader("进化版本与防退化门控")
    if not version_records:
        st.info("尚未生成 skill_versions.jsonl。运行 evolve-demo 后会出现版本轨迹。")
        return

    rows = []
    for record in version_records:
        metrics = record.get("metrics", {}) or {}
        rows.append(
            {
                "ts": record.get("ts"),
                "event": record.get("event"),
                "evolution_action": record.get("evolution_action"),
                "target_category": record.get("target_category"),
                "root_cause_type": record.get("root_cause_type"),
                "confidence": record.get("confidence"),
                "baseline_f1": metrics.get("baseline_f1"),
                "sample_f1": metrics.get("sample_f1"),
                "replay_f1": metrics.get("replay_f1"),
                "full_f1": metrics.get("full_f1"),
                "skill_count": metrics.get("skill_count"),
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df.tail(50), use_container_width=True, hide_index=True)

    event_counts = df["event"].value_counts().reset_index()
    event_counts.columns = ["event", "count"]
    st.plotly_chart(px.bar(event_counts, x="event", y="count", title="版本事件分布"), use_container_width=True)

    f1_cols = [col for col in ["baseline_f1", "sample_f1", "replay_f1", "full_f1"] if col in df.columns]
    f1_df = df[f1_cols].dropna(how="all")
    if not f1_df.empty:
        f1_df = f1_df.reset_index(names="step")
        st.plotly_chart(px.line(f1_df, x="step", y=f1_cols, markers=True, title="三闸门验证结果"), use_container_width=True)


def render_skill_repo(skill_text):
    st.subheader("SkillRepo 状态")
    skill_df = parse_skill_blocks(skill_text)
    if skill_df.empty:
        st.info("尚未生成结构化 Skill。")
        return

    total_chars = int(skill_df["chars"].sum())
    total_tokens = int(skill_df["approx_tokens"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Skill 数量", len(skill_df))
    c2.metric("字符数", total_chars)
    c3.metric("估算 Token", total_tokens)

    counts = skill_df.groupby("category").size().reset_index(name="count")
    st.plotly_chart(px.bar(counts, x="category", y="count", title="按类别统计 Skill 数"), use_container_width=True)
    st.dataframe(skill_df[["category", "title", "chars", "approx_tokens"]], use_container_width=True, hide_index=True)

    with st.expander("查看 SKILL.md"):
        st.markdown(skill_text)


def render_tip_memory(tip_records):
    st.subheader("双层记忆：TipMemory -> 长期资产")
    if not tip_records:
        st.info("尚未生成 tips.jsonl。运行 evolve-demo 或开启进化后会先沉淀短期经验。")
        return

    rows = []
    for record in tip_records:
        rows.append(
            {
                "tip_id": record.get("tip_id"),
                "status": record.get("status"),
                "count": record.get("count"),
                "root_cause_type": record.get("root_cause_type"),
                "evolution_action": record.get("evolution_action"),
                "target_category": record.get("target_category"),
                "confidence": record.get("confidence"),
                "first_ts": record.get("first_ts"),
                "last_ts": record.get("last_ts"),
                "proposed_rule": str(record.get("proposed_rule") or "")[:120],
            }
        )
    df = pd.DataFrame(rows)
    promoted = int((df["status"] == "promoted").sum())
    buffered = int((df["status"] == "buffered").sum())
    rejected = int((df["status"] == "rejected").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tip 总数", len(df))
    c2.metric("已晋升", promoted)
    c3.metric("缓冲中", buffered)
    c4.metric("已拒绝", rejected)

    status_counts = df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    st.plotly_chart(px.bar(status_counts, x="status", y="count", title="短期经验状态"), use_container_width=True)

    action_counts = df.groupby(["evolution_action", "status"], dropna=False).size().reset_index(name="count")
    st.plotly_chart(px.bar(action_counts, x="evolution_action", y="count", color="status", title="按进化动作统计"), use_container_width=True)
    st.dataframe(df.tail(100), use_container_width=True, hide_index=True)


def render_saf_traces(trace_records):
    st.subheader("State-Action-Feedback 轨迹")
    if not trace_records:
        st.info("尚未生成 saf_traces.jsonl。运行 benchmark 或 evolve-demo 后会出现统一轨迹。")
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
                "input": str(state.get("raw_input", ""))[:80],
            }
        )
    df = pd.DataFrame(rows)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trace 数", len(df))
    c2.metric("平均反馈分", f"{pd.to_numeric(df['score'], errors='coerce').mean():.2f}")
    c3.metric("Exact Match", f"{pd.to_numeric(df['exact_match'], errors='coerce').mean():.2%}")

    grouped = df.groupby(["domain", "modality"], dropna=False).size().reset_index(name="count")
    st.plotly_chart(px.bar(grouped, x="domain", y="count", color="modality", title="跨领域/模态轨迹分布"), use_container_width=True)
    st.dataframe(df.tail(100), use_container_width=True, hide_index=True)


def render_latest_patch(patch):
    st.subheader("最新归因与优化指令")
    if not patch:
        st.info("尚未生成 latest_patch.json。")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("错误类型", str(patch.get("root_cause_type") or patch.get("error_category") or "-"))
    c2.metric("进化动作", str(patch.get("evolution_action") or "-"))
    c3.metric("置信度", f"{float(patch.get('confidence') or 0):.2f}")
    st.caption(f"目标类别: {patch.get('target_category') or '-'}")
    st.write("根因分析")
    st.write(patch.get("root_cause_analysis") or "-")
    st.write("建议规则")
    st.code(patch.get("proposed_rule") or "-", language="markdown")
    with st.expander("完整 patch JSON"):
        st.json(patch)


st.title("Self-Evolving Harness Dashboard")
st.caption("展示执行、评估、反思、进化、防退化与跨领域 State-Action-Feedback 轨迹。")

metrics = read_csv("metrics.csv")
tokens = read_csv("token_usage.csv")
versions = read_jsonl("skill_versions.jsonl")
traces = read_jsonl("saf_traces.jsonl", limit=3000)
tips = read_jsonl("tips.jsonl", limit=3000)
skill_text = read_text("SKILL.md")
latest_patch = read_json("latest_patch.json")

tab_overview, tab_evolution, tab_skill, tab_tip, tab_trace, tab_patch = st.tabs(
    ["总览", "进化与回滚", "SkillRepo", "双层记忆", "SAF 轨迹", "最新归因"]
)

with tab_overview:
    render_overview(metrics, tokens)

with tab_evolution:
    render_evolution(versions)

with tab_skill:
    render_skill_repo(skill_text)

with tab_tip:
    render_tip_memory(tips)

with tab_trace:
    render_saf_traces(traces)

with tab_patch:
    render_latest_patch(latest_patch)

if st.button("刷新数据"):
    st.cache_data.clear()
    st.rerun()
