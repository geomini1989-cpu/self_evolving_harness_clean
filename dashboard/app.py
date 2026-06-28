import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px

# 设置页面配置
st.set_page_config(layout="wide", page_title="Self-Evolving Harness Console")

st.title("🚀 Self-Evolving Harness 实时进化控制台")

# 1. 进化指标折线图区域
st.subheader("📊 核心性能指标监控 (F1 Score)")
# 这里我们假设有一个 metrics.csv 记录了每次 Epoch 的结果
if os.path.exists("memory/metrics.csv"):
    df = pd.read_csv("memory/metrics.csv")
    fig = px.line(df, x="epoch", y="f1_score", markers=True, title="模型准确率进化曲线")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("暂无进化数据，请先启动 main_loop.py")

# 2. 布局：左侧知识库，右侧实时诊断
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 动态演进中的技能库 (SKILL.md)")
    if os.path.exists("memory/SKILL.md"):
        with open("memory/SKILL.md", "r", encoding="utf-8") as f:
            skill_content = f.read()
            st.markdown(skill_content)
    else:
        st.warning("SKILL.md 尚未生成")

with col2:
    st.subheader("🔍 最新归因诊断与补丁")
    # 这里我们展示最近一次的诊断记录
    if os.path.exists("memory/latest_patch.json"):
        with open("memory/latest_patch.json", "r", encoding="utf-8") as f:
            patch = json.load(f)
            st.success(f"错误分类: {patch.get('error_category')}")
            st.write(f"**根因分析**: {patch.get('root_cause_analysis')}")
            st.code(f"{patch.get('proposed_rule')}", language="markdown")
    else:
        st.write("等待系统捕捉第一个 Bad Case...")

# 实时刷新按钮
if st.button("刷新控制台"):
    st.rerun()