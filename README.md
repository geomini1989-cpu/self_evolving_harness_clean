# Self-Evolving Harness

面向技术比武题目一“算法方向”的自进化智能体评测与优化平台。项目以客户投诉工单结构化抽取为主任务，冻结基座大模型参数，通过批量执行、自动评估、根因反思、Prompt/Skill/Few-shot 进化、经验回放和版本回滚，在低 Token 成本下提升或稳定 F1 指标。

## 项目定位

本项目不是微调模型，也不依赖梯度更新。核心目标是验证一种可迁移的 Harness 框架：把 Agent 的运行过程抽象为“状态 -> 动作 -> 反馈”的闭环，然后在外层框架中持续优化策略、记忆和执行约束。

对应比赛要求：

- 可视化与交互界面：提供 Streamlit Dashboard，展示 F1、Token、缓存、进化事件、记忆资产和跨领域迁移结果。
- 模型与 API 资源：兼容 DashScope / OpenAI-compatible Chat Completions API；无 API_KEY 时可进入离线演示模式。
- 数据集：默认读取本地 `data/dataset.csv`，支持抽样和全量 benchmark。
- 自迭代闭环引擎：内置“执行 -> 评估 -> 反思 -> 进化 -> 回归门控”的自动化流水线。
- 防退化机制：每次候选更新都经过置信度门控、bad-case 小样本验证、经验回放和必要时全量回归；不达标自动回滚。
- 跨领域迁移：通过统一 SAF 抽象和 `tools/transfer_test_runner.py` 演示文本、图像元信息、语音转写、推荐流等同类任务的迁移能力。

## 创新点

### 1. 冻结模型下的外部自进化框架

传统方案往往依赖微调、RAG 堆上下文或人工修改 Prompt。本项目把优化能力放在模型外层：基座模型参数保持冻结，由 Harness 自动完成执行、评估、反思、进化和回归。这样既符合小算力约束，也能清楚证明指标提升来自框架策略，而不是重新训练模型。

### 2. 双层记忆机制

项目把记忆分为短期和长期两层：

- 短期记忆：`optimizer/tip_memory.py` 先缓存 bad case 修复经验，避免一次偶然失败就污染长期规则。
- 长期记忆：只有高置信或重复出现的经验，才晋升为 `memory/SKILL.md`、`memory/PROMPT_POLICY.md` 或 `memory/examples.json`。

这种设计兼顾学习速度和安全性，能体现“系统持续进化”，也能避免 Skill 盲目膨胀。

### 3. Prompt / Skill / Few-shot 联动进化

进化动作不是单一追加 Prompt，而是根据根因自动选择：

- `prompt_patch`：适合 schema 约束、输出格式、字段解释类问题。
- `skill_patch`：适合稳定业务知识、类别边界和领域规则。
- `few_shot_patch`：适合局部样例、低风险修复和相似 case 迁移。

这比单纯维护一个大 Prompt 更细粒度，也更容易做回滚和成本控制。

### 4. 防退化闭环

每个候选补丁写入后不会立即永久生效，而是进入回归门控：

1. 先验证当前 bad-case 相关样本。
2. 再验证 replay set，防止修复局部破坏全局。
3. 风险较高或接近阈值时再触发更大范围回归。
4. F1 下降超过 `f1_tolerance` 自动回滚。

这对应比赛要求中的“避免灾难性遗忘”和“版本回滚”。

### 5. Token-Aware Harness

本项目不是简单调用 LLM，而是把 Token 成本作为一等指标来优化：

- 本地类别路由替代 LLM scout。
- 按类别检索 Skill，避免全量注入。
- 默认关闭 Few-shot，只在必要时启用。
- 对确定性调用做磁盘缓存。
- Dashboard 直接展示调用数、Token、缓存命中和耗时。

因此系统既能展示 F1，也能展示成本曲线。

### 6. SAF 跨领域统一抽象

通过 `adapters/saf.py`，不同任务被统一为：

```text
State -> Action -> Feedback
```

主循环不绑定某一种数据模态，领域差异由 adapter 承担。`tools/transfer_test_runner.py` 已用文本、图像元信息、语音转写和推荐流做了轻量迁移验证，体现“技术方案可快速迁移到同类任务场景”。

## 优化点

### 1. Token 消耗优化

已完成的 Token 优化包括：

- 移除批处理前的 LLM scout 调用，减少每轮固定额外请求。
- 使用 `MemoryBank.route_categories()` 做本地关键词/正则路由。
- 使用 `MemoryBank.get_skills_by_categories()` 只注入命中类别 Skill。
- Prompt 改为短字段协议和紧凑 JSON 数组输出。
- `BaseLLMClient` 增加磁盘缓存，key 包含模型、温度、max_tokens 和 prompt hash。
- cache hit 时直接复用历史结果，并记录为 0 Token。
- Few-shot 默认关闭，避免每个 batch 重复注入样例。
- `benchmark` 模式提供本地 fast path，可用于全量低成本指标报告。

### 2. 批处理延迟优化

已完成的延迟优化包括：

- `batch_size`、`max_workers`、`epochs`、`sample_size` 全部命令行可配置。
- 批量请求替代逐条请求，减少 API 往返。
- 并发执行 batch，提高吞吐。
- JSON 解析失败时自动拆小批重试，而不是整轮失败。
- 默认 benchmark 只跑 1 轮，适合快速评估全量数据。
- Dashboard 记录每轮 `elapsed_ms`，便于对比优化前后耗时。

### 3. F1 指标优化

已完成的 F1 优化包括：

- 字段级 F1 评估，能精细定位 `core_intent`、`urgency_level`、`entities`、`summary` 的错误。
- `F1PostProcessor` 做 schema 修复、紧急程度纠正、实体补全和摘要裁剪。
- bad case 自动进入归因流程，生成可执行补丁。
- 对局部错误优先采用 Few-shot 修复，降低破坏全局策略的风险。
- 对稳定类别边界错误沉淀为 Skill，提升后续同类样本表现。

### 4. 自进化可靠性优化

已完成的可靠性优化包括：

- 置信度门控：低置信补丁不写入长期记忆。
- 高风险识别：涉及广泛 schema 或多字段冲突的补丁会被更严格审查。
- rejected buffer：重复失败的候选不再反复尝试。
- SkillRepo 治理：限制每类 Skill 数量，做去重和淘汰。
- 版本日志：所有 accepted、rejected、rolled_back、curated 事件写入 `skill_versions.jsonl`。
- 回滚机制：补丁导致样本或回放 F1 下降时自动恢复旧资产。

### 5. 可观测性优化

已完成的观测优化包括：

- `memory/metrics.csv`：记录每轮 F1、调用数、Token、缓存命中、耗时。
- `memory/token_usage.csv`：记录每次 LLM 调用的 prompt/completion/total token。
- `memory/saf_traces.jsonl`：记录状态、动作、反馈轨迹。
- Streamlit Dashboard 汇总曲线、事件分布、记忆资产和迁移结果。
- README 提供低成本演示、正式演示和全量 benchmark 三套命令。

### 6. 工程可维护性优化

已完成的工程优化包括：

- 主循环、LLM 客户端、评估器、记忆库、归因器、进化器模块化拆分。
- 配置项集中在 `adapters/ticket_config.yaml`，命令行可覆盖。
- `.env.example` 移除真实 key，仅保留占位符。
- 运行态 memory 与代码分离，便于选择“从 0 演示”或“保留学习成果演示”。
- 跨领域 Transfer Runner 独立在 `tools/` 下，不污染主任务逻辑。

## 论文启发

实现思路参考以下方向，并落到工程组件中：

- SkillOS：把长期技能沉淀到 `memory/SKILL.md`，按业务类别检索命中技能，避免每轮注入整份 Skill。
- SkillOpt：在模型输出后增加轻量策略层，做 schema 修复、置信度门控和 F1 友好的后处理。
- Self-Harness：将执行、评估、反思、进化纳入统一闭环，由 Harness 驱动模型外部能力增长。
- SkillClaw / SkillAdaptor：对 Skill 做版本管理、去重淘汰、风险控制，并支持从轨迹中提炼可迁移经验。
- Memento-Skill：保留成功样例和失败修复经验，让后续运行复用历史上下文。

## 核心能力

### 1. 低 Token 批处理

- 移除批处理前的 LLM scout 调用，改为本地关键词/正则路由。
- 只注入命中类别相关 Skill，而不是全量 Skill 文档。
- 默认关闭 Few-shot，仅在进化或回放需要时少量注入。
- 持久化确定性 LLM 缓存，跨运行复用同 prompt 结果。
- benchmark 模式默认使用本地规则 fast path，用于快速报告全量 F1 和展示 Token 节省。

### 2. 自动化自进化闭环

每轮运行包含以下步骤：

1. 执行：按 batch 调用 LLM 或本地 fast path，输出紧凑 JSON。
2. 评估：对预测和 ground truth 计算字段级 F1。
3. 反思：发现 bad case 后由 LLM 或规则归因器判断根因。
4. 进化：生成 `prompt_patch`、`skill_patch` 或 `few_shot_patch`。
5. 验证：先跑相关 bad-case 小样本，再跑经验回放集。
6. 回滚：若 F1 下降超过阈值，自动恢复旧版本，并写入 rejected buffer。

### 3. 防灾难性遗忘

- `memory/examples.json`：Few-shot 和经验回放样本，最多保留最近 100 条。
- `memory/rejected_skills.jsonl`：被拒绝候选缓存，避免重复尝试同一坏更新。
- `memory/skill_versions.jsonl`：记录接受、拒绝、回滚、整理 SkillRepo 等事件。
- `memory/PROMPT_POLICY.md`：保存被接受的 Prompt 策略补丁。
- `memory/SKILL.md`：保存被接受的长期业务 Skill。
- `optimizer/tip_memory.py`：短期经验先进入缓冲区，重复出现或高置信后才晋升为长期资产。

### 4. 可视化 Dashboard

Dashboard 展示：

- F1 演化曲线
- Token 消耗和缓存命中
- 每轮调用数与耗时
- 进化事件分布
- 已接受 / 回滚 / 拒绝的 Skill 和 Prompt 资产
- Few-shot 样例
- SAF 轨迹和跨领域迁移测试结果

## 项目结构

```text
self_evolving_harness_clean/
├── adapters/
│   ├── saf.py                 # 状态-动作-反馈统一抽象
│   ├── ticket_adapter.py      # 客诉工单领域适配器
│   └── ticket_config.yaml     # 任务、运行和进化配置
├── core/
│   ├── evaluator.py           # 字段级评估与 F1 计算
│   ├── f1_optimizer.py        # SkillOpt 风格输出后处理
│   ├── llm_client.py          # LLM 调用、离线模式、磁盘缓存、Token 记录
│   └── memory_bank.py         # Skill/Few-shot 读取、类别路由和检索
├── optimizer/
│   ├── attributor.py          # bad-case 根因归因
│   ├── evolver.py             # Prompt/Skill/Few-shot 更新、回归和回滚
│   └── tip_memory.py          # 双层记忆中的短期经验缓冲
├── dashboard/
│   └── app.py                 # Streamlit 演示驾驶舱
├── tools/
│   └── transfer_test_runner.py# 跨领域迁移测试
├── data/
│   └── dataset.csv            # 本地评测数据
├── memory/                    # 运行时记忆、缓存和指标
├── main_loop.py               # 主入口
├── requirements.txt
└── README.md
```

## 环境准备

建议使用 Windows PowerShell，在项目根目录执行：

```powershell
cd D:\self_evolving_harness_clean
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

依赖很轻：

```text
openai
datasets
streamlit
plotly
```

## API 配置

复制模板：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```text
API_KEY=你的 DashScope 或 OpenAI-compatible API Key
LLM_TIMEOUT_SECONDS=90
LLM_OFFLINE=0
```

如果 `API_KEY` 为空，系统会提示：

```text
[LLM] API_KEY not found. Running in offline demo mode.
```

这不是报错，而是离线演示模式。离线模式会使用本地规则模拟 LLM 输出，适合快速展示流程；真实多轮自进化需要填入有效 API_KEY。

注意：不要提交 `.env`。`.env.example` 只应保留占位符。

如果只想做本地 smoke test、不消耗 API Token，可以临时设置：

```powershell
$env:LLM_OFFLINE="1"
python main_loop.py --mode demo --sample-size 8 --epochs 1 --batch-size 4 --max-workers 1 --no-evolution
```

## 运行模式

### 1. 快速 Demo

适合确认项目能跑通：

```powershell
python main_loop.py --mode demo
```

默认抽取少量样本，多轮执行，并开启自进化闭环。

### 2. 真实 LLM 自进化演示

适合答辩展示“从 0 开始，执行-评估-反思-进化”：

```powershell
python main_loop.py --mode llm-evolve --sample-size 300 --epochs 6 --batch-size 8 --max-workers 2 --reset-state
```

参数含义：

- `--sample-size 300`：抽取 300 条样本参与演示。
- `--epochs 6`：运行 6 轮，便于观察曲线和进化事件。
- `--batch-size 8`：每次 LLM 请求处理 8 条，控制 JSON 稳定性和 Token。
- `--max-workers 2`：限制并发，减少 API 压力。
- `--reset-state`：清空运行态 memory/cache，从 0 展示自进化。

如果想更省 Token，可先用：

```powershell
python main_loop.py --mode llm-evolve --sample-size 100 --epochs 3 --batch-size 8 --max-workers 2 --reset-state
```

### 3. 全量 Benchmark

适合报告当前数据集总体 F1：

```powershell
python main_loop.py --mode benchmark
```

benchmark 模式默认：

- 跑全量数据。
- 只跑 1 轮。
- 关闭进化。
- 使用本地 skill fast path，LLM 调用数接近 0。

如果你希望 benchmark 也调用真实 LLM：

```powershell
python main_loop.py --mode benchmark --llm-benchmark --sample-size 1000
```

真实 LLM benchmark 会明显增加 Token 和耗时，建议先小样本验证。

### 4. 强制进化演示

适合在没有自然 bad case 时强制触发一次演化流程：

```powershell
python main_loop.py --mode evolve-demo --use-llm-evolution
```

## 常用参数

| 参数 | 说明 |
| --- | --- |
| `--mode` | `demo`、`benchmark`、`evolve-demo`、`llm-evolve` |
| `--sample-size` | 评测样本数；可填数字或 `all` |
| `--epochs` | 运行轮数 |
| `--batch-size` | 每个 LLM batch 的样本数 |
| `--max-workers` | 并发 batch 数 |
| `--reset-state` | 运行前清空 memory 和 cache |
| `--no-evolution` | 发现 bad case 后不执行进化 |
| `--llm-benchmark` | benchmark 模式改用真实 LLM |
| `--use-llm-evolution` | 使用 LLM 做归因和补丁生成 |
| `--force-evolution-demo` | 强制触发一次进化演示 |

## Dashboard 使用

启动：

```powershell
streamlit run dashboard\app.py
```

默认访问：

```text
http://localhost:8501
```

如果端口被占用，可指定端口：

```powershell
streamlit run dashboard\app.py --server.port 8502
```

建议演示顺序：

1. 先运行一次 `llm-evolve` 生成多轮指标和 memory。
2. 打开 Dashboard。
3. 查看“总览”中的 F1、Token、缓存命中和进化事件。
4. 查看“记忆资产”中的 Skill、Few-shot、Prompt Policy。
5. 查看“回归防线”中的 rejected / rolled back 记录。
6. 查看“迁移评估”证明框架可迁移。

## Runtime 产物说明

运行后会在 `memory/` 下生成或更新：

| 文件 | 作用 |
| --- | --- |
| `metrics.csv` | 每轮 F1、调用数、缓存命中、Token、耗时 |
| `token_usage.csv` | 每次 LLM 调用的输入/输出/总 Token |
| `llm_cache.jsonl` | 确定性 LLM 调用磁盘缓存 |
| `examples.json` | Few-shot 样例和经验回放样本 |
| `SKILL.md` | 被接受的长期 Skill 文档 |
| `PROMPT_POLICY.md` | 被接受的 Prompt 策略补丁 |
| `skill_versions.jsonl` | 进化事件、接受、拒绝、回滚和整理记录 |
| `rejected_skills.jsonl` | 被拒绝候选，防止重复坏更新 |
| `latest_patch.json` | 最近一次候选补丁 |
| `saf_traces.jsonl` | 状态-动作-反馈轨迹 |
| `transfer_report.json` | 跨领域迁移测试汇总 |
| `transfer_traces.jsonl` | 跨领域迁移轨迹 |

这些文件体现了自进化过程。演示“从 0 开始”时使用 `--reset-state`；日常优化时不要清空，系统会复用历史经验并避免重复 Token 消耗。

## 配置项

主要配置在 `adapters/ticket_config.yaml`：

```yaml
runtime:
  mode: demo
  batch_size: 8
  max_workers: 6
  epochs: 6
  sample_size: 30
  enable_few_shots: false
  enable_llm_scout: false
  enable_f1_postprocess: true
  enable_rule_fast_path: false
  enable_evolution: true

cache:
  enabled: true
  path: memory/llm_cache.jsonl

evolution:
  regression_mode: sample_then_full
  f1_tolerance: 0.02
  confidence_threshold: 0.70
  sample_size: 10
  replay_size: 20
  max_new_skill_chars: 900
  max_skills_per_category: 5
```

命令行参数会覆盖配置文件中的运行参数。

## 跨领域迁移测试

运行：

```powershell
python tools\transfer_test_runner.py
```

只跑某个 suite：

```powershell
python tools\transfer_test_runner.py --suite visual_quality
```

当前 Transfer Runner 覆盖：

- `ticket_text`：客户投诉文本结构化。
- `visual_quality`：图像元信息质检。
- `speech_intent`：语音转写意图识别。
- `recommendation_stream`：推荐事件流状态判断。

输出：

- `memory/transfer_report.json`
- `memory/transfer_traces.jsonl`

这部分用于说明 Harness 的底层架构不是只服务客诉任务，而是可以通过 adapter 快速迁移到同类任务。

## 推荐演示方案

### 低成本流程演示

```powershell
python main_loop.py --mode llm-evolve --sample-size 100 --epochs 3 --batch-size 8 --max-workers 2 --reset-state
streamlit run dashboard\app.py
```

优点：Token 成本可控，能展示闭环、缓存、回滚和记忆。

### 正式答辩演示

```powershell
python main_loop.py --mode llm-evolve --sample-size 300 --epochs 6 --batch-size 8 --max-workers 2 --reset-state
python tools\transfer_test_runner.py
streamlit run dashboard\app.py
```

优点：样本和轮数更充分，Dashboard 曲线更完整，也能展示迁移能力。

### 全量指标报告

```powershell
python main_loop.py --mode benchmark
```

优点：快速得到全量 F1；默认 0 Token 或近 0 Token，适合最终汇报吞吐和成本。

## 如何判断效果是否好

建议同时看三类指标：

- 准确率：`metrics.csv` 中的 `f1_score` 是否稳定或提升。
- 成本：`calls`、`total_tokens`、`cache_hits` 是否符合预期。
- 稳定性：`skill_versions.jsonl` 中是否有 accepted、rolled_back、rejected 记录，证明系统不是盲目写入，而是有防退化门控。

如果 F1 某轮下降，不一定是坏事。关键是候选补丁是否被回滚，长期记忆是否没有被污染。这个项目更强调“安全进化”，不是每轮都强行让曲线上升。

## 常见问题

### 1. 为什么 benchmark 是 0 Token？

benchmark 默认走本地 skill fast path，用于快速报告全量评测指标和展示框架节省 Token 的能力。如果要真实调用模型，请加 `--llm-benchmark`。

### 2. 为什么没有生成 `SKILL.md`？

只有候选 `skill_patch` 通过置信度、样本回归和经验回放后，才会写入 `memory/SKILL.md`。如果补丁被回滚或策略选择了 Few-shot 优先，可能只更新 `examples.json` 或记录到 `rejected_skills.jsonl`。

### 3. 为什么多轮 F1 不一定一直上升？

LLM 输出存在波动，且系统会优先防止全局退化。某些局部修复如果伤害 replay set，会被回滚。因此曲线可能小幅波动，但长期记忆应保持安全。

### 4. 每次运行是否要清空 memory？

演示从 0 自进化时使用 `--reset-state`。真实优化不建议每次清空，因为缓存、Few-shot、Skill 和 rejected buffer 都是降低 Token 和防止重复试错的重要资产。

### 5. Skill 会不会越来越大浪费 Token？

系统有两层控制：

- 检索控制：每次只注入命中类别的少量 Skill。
- 仓库治理：`max_skills_per_category` 限制每类 Skill 数量，并对重复或低收益候选做拒绝/回滚。

## Git 建议

建议提交代码和配置：

- `main_loop.py`
- `adapters/`
- `core/`
- `optimizer/`
- `dashboard/`
- `tools/`
- `requirements.txt`
- `README.md`
- `.env.example`

不建议提交：

- `.env`
- `.venv/`
- `memory/llm_cache.jsonl`
- `memory/token_usage.csv`
- 大量临时运行日志

`memory/SKILL.md`、`memory/PROMPT_POLICY.md`、`memory/examples.json` 是否提交取决于演示策略：如果要展示“已经学习到的能力”，可以挑选稳定版本提交；如果要展示“从 0 开始进化”，就不要提交这些运行态资产。

## 一句话总结

Self-Evolving Harness 通过“低 Token 批处理 + 双层记忆 + 根因归因 + 安全进化 + 回归回滚 + 可视化观测”，在不训练模型的前提下，让 Agent 具备可验证、可回退、可迁移的持续优化能力。
