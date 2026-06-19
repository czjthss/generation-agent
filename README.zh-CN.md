# generation-agent

[English](README.md) | [中文](README.zh-CN.md)

一个基于 MCP 工具形态和 LangChain Agent 的时间序列生成智能体。输入一句自然语言描述，例如：

```text
生成中国南方夏季工业园区的电力负载
```

输出为生成完成的时间序列 Arrow 文件。

## 结构

- `generation_agent/features.py`: 通用时间序列特征生成器，包括周期、趋势、热效应、噪声、异常。
- `generation_agent/planner.py`: 将一句话描述转换为生成计划。计划会包含 `domain` 和 `generator_type`，用于选择领域生成机制。设置 `OPENAI_API_KEY` 后使用 LangChain 多 Agent 调用特征工具；缺少 LLM 配置时直接报错。
- `generation_agent/planning_prompts.py`: 定义角色化 Planning、Reflection 和直接 JSON 规划提示词。
- `generation_agent/specialist_workflow.py`: 面向生成任务的需求规格、过程设计、领域反证角色。
- `generation_agent/langchain_agent.py`: LangChain 工具调用工作流。Planning 角色将领域知识映射为机制和参数；Reflection 角色独立审查计划，发现语义或约束冲突时触发一次重规划。
- `generation_agent/feature_composer.py`: 将 LLM 选择的特征参数组合成 `SeriesPlan`。
- `generation_agent/domain_rules.py`: 保存和复用 LLM 新增的自定义领域规则，默认写入项目根目录 `domain_rules.json`。
- `generation_agent/synthesizer.py`: 根据计划合成时间序列。不同领域不会全部套用三角/正弦周期，而是使用领域机制。
- `generation_agent/semantic_transforms.py`: 执行累计、存量流量、状态切换、随机游走、衰减恢复、饱和增长和多变量滞后机制。
- `generation_agent/semantic_validators.py`: 校验单调性、上下界、整数性、累计恒等式和存量守恒。
- `generation_agent/semantic_types.py`: 定义异常运行时覆盖接口和强度预设。
- `generation_agent/codegen.py`: 生成可独立运行的 Python 代码。
- `generation_agent/mcp_server.py`: MCP server，暴露 `plan_time_series`、`generate_time_series`、`generate_time_series_code` 等工具。
- `generation_agent/cli.py`: 命令行入口。

## 快速运行

在项目目录中：

```bash
python -m generation_agent.cli "生成中国南方夏季工业园区的电力负载" \
  --length 168 \
  --output outputs/load.arrow
```

## 单序列与数据集模式

原有单序列模式保持不变：

```bash
python -m generation_agent.cli "生成北京市夏季降水" \
  --generation-mode sequence \
  --length 720 \
  --output outputs/beijing_rain.arrow
```

数据集模式输入一个领域，由 `Dataset Scenario Designer` 先生成互不重复的具体描述，再复用后续规划和生成链路：

```bash
python -m generation_agent.cli "气象" \
  --generation-mode dataset \
  --series-count 20 \
  --length 720 \
  --output-dir outputs/weather_dataset
```

目录中包含 `series_*.arrow`、`scenarios.json` 和包含计划及校验信息的 `manifest.json`。

## 参考序列

参考序列由确定性代码提取分布、趋势、自相关、周期候选、稀疏性和事件持续时间，再由 `Reference Interpreter` 转换成生成先验。原始观测值不会交给 LLM，也不会被复制。

```bash
python -m generation_agent.cli "生成同类工业负载" \
  --reference reference_load.arrow \
  --length 1000 \
  --output outputs/synthetic_load.arrow
```

时间列和值列会自动识别，参考策略固定为 `structure`：参考尺度、分布、趋势、动态结构、周期候选和稀疏性，同时拒绝与目标领域语义冲突的特征。

降水量这类间歇事件会生成大量 0 和少量雨段，而不是每天都有平滑周期：

```bash
python -m generation_agent.cli "生成中国南方夏季降水量数据" \
  --length 240 \
  --output outputs/rain.arrow
```

已内置的主要生成机制：

- `cyclic_signal`: 电力负载等有日周期/周周期的数据。
- `intermittent_event`: 降水、暴雨等稀疏事件过程，先生成雨段，再生成雨强。
- `daylight_envelope`: 光伏发电，夜间为 0，白天按日照包络变化，并可受云遮挡影响。
- `smooth_environmental`: 温度等带惯性的环境变量。
- `count_process`: 交通流量、订单量、API 请求量等非负计数数据。
- `bounded_utilization`: CPU/内存等有上下界的利用率数据。

基础生成器之上增加了 8 类输出语义：

- `instantaneous`: 普通瞬时值。
- `cumulative`: 先生成每期增量，再累计求和；适用于累计销售额、累计里程。
- `stock_flow`: 按“上期存量 + 流入 - 流出”递推；适用于库存、余额和水量。
- `regime_switching`: 在开机、停机、故障等状态之间切换。
- `random_walk`: 使用漂移和随机步长递推；适用于价格、汇率等。
- `decay_recovery`: 冲击后按递推关系衰减或恢复。
- `saturation_growth`: 使用 Logistic 机制向容量上限增长。
- `multivariate_lag`: 生成驱动变量及其滞后影响。

LLM 会同时选择 `generator_type` 和 `semantic_type`。前者决定基础增量或驱动信号如何产生，后者决定最终输出必须遵循的数学机制。

语义机制按统一流程执行：LLM 根据观测量的数学定义选择机制和约束，确定性代码完成变换、异常注入、校验与修复。任何语义类型都不在 System Prompt 中拥有领域特例。

这些工具不是按领域穷举，因此不可能也不需要覆盖“所有领域名称”。它们覆盖的是常见时间序列生成特征族：

- 周期/趋势/季节性
- 稀疏事件和持续事件
- 日照或工作窗口约束
- 平滑惯性过程
- 非负计数过程
- 有上下界的利用率过程
- 噪声和异常

LLM 会先识别领域，再把领域知识映射到这些特征参数。对于新领域，LLM 可以创建自定义规则供后续复用。

当前 LangChain tools 只保留特征/规则工具：

- `inspect_feature_coverage`: 检查当前特征生成器能否表达该请求。
- `finalize_feature_plan`: 由 LLM 传入领域、生成机制和全部特征参数，生成最终计划。
- `use_existing_custom_domain_rule`: 复用之前保存的新领域规则。
- `create_custom_domain_rule`: 将新领域映射到特征参数并保存为规则。

其中异常注入参数也由 LLM 根据领域语义传入：

- `anomaly_enabled`
- `anomaly_severity`
- `anomaly_target`
- `anomaly_count`
- `anomaly_kind`
- `anomaly_magnitude`
- `anomaly_width`

例如提示中出现尖峰、突降、故障、断电、开门、拥塞、暴雨等语义时，LLM 会把这些语义映射为对应异常参数。异常会注入到语义正确的位置，例如累计量的增量、库存的流入/流出、随机游走的步长或状态切换的状态，而不总是直接修改最终值。

异常控制接口：

```bash
# 采用 LLM 的判断
python -m generation_agent.cli "生成工业园区电力负载" --anomalies auto

# 完全禁用异常
python -m generation_agent.cli "生成工业园区电力负载" --anomalies off

# 强制启用高强度异常
python -m generation_agent.cli "生成累计销售额" \
  --anomalies on \
  --anomaly-severity high
```

强度预设：

- `low`: 少量、短持续、较小幅度。
- `medium`: 中等数量、幅度和持续时间。
- `high`: 更多、更强、持续更久。

异常数量、幅度、持续时间、类型和作用位置由 LLM 根据领域语义决定；注入方向固定同时允许正向和负向偏离。

Python API：

```python
from generation_agent import AnomalyOverrides, GenerationAgent

agent = GenerationAgent()
df = agent.generate(
    "生成累计销售额",
    length=168,
    anomaly_overrides=AnomalyOverrides(
        enabled=True,
        severity="high",
    ),
)
```

如需使用 LangChain Agent 工具调用规划：

```bash
python -m generation_agent.cli "生成中国南方夏季工业园区的电力负载，包含偶发尖峰异常" \
  --length 336 \
  --output outputs/load.arrow
```

推荐在项目根目录创建 `.env`：

```bash
cp .env.example .env
```

然后编辑 `.env`：

```text
OPENAI_API_KEY=你的 key
OPENAI_BASE_URL=https://api.ofox.ai/v1
GENERATION_AGENT_MODEL=gpt-5.5
```

代码会自动读取项目根目录 `.env`。已经在 shell 中设置的环境变量优先级更高，不会被 `.env` 覆盖。

代码默认使用 OpenAI 兼容接口：

```text
https://api.ofox.ai/v1
```

如需切换到其他兼容接口，可以设置：

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

CLI 默认使用 `gpt-5.5` 进行规划；也可以显式指定：

```bash
python -m generation_agent.cli "生成中国南方夏季降水量数据" \
  --model gpt-5.5 \
  --length 240 \
  --output outputs/rain.arrow
```

## 角色化工作流

设计参考 [CastFlow](https://github.com/Forever-Pan/CastFlow) 的角色专门化思想，并针对“从描述生成数据”的任务做了适配：

```text
用户描述
  -> Planning（领域分析、机制设计、异常设计、约束自检）
  -> LangChain 特征工具生成 SeriesPlan
  -> Reflection 独立审查计划
       -> REVISE：携带问题反馈重规划一次
       -> PASS：进入执行
  -> 确定性数值合成与语义变换
  -> 异常注入
  -> 约束校验与修复
  -> Arrow
```

Planning Prompt 不再列举“某个领域必须怎样生成”的案例，而是要求 LLM 依次判断观测量、基础信号、演化语义、异常作用位置和数学约束。这样可以降低对关键词和少数示例的过拟合。已有自定义规则作为策略记忆使用，但只在观测量和上下文明确匹配时复用。

借鉴 [NEXUS](https://arxiv.org/abs/2605.14389) 的“专门化推理后再综合”原则，但工作流按生成任务重新设计：

```text
Specification Agent
  -> Process Architect
  -> Domain Challenger
  -> Plan Compiler（LangChain tools）
  -> Reflection Agent
```

- `Specification Agent`: 定义一个数据点究竟表示什么，包括单位、时间口径、取值空间和不变量。
- `Process Architect`: 设计随机过程、事件机制、时间依赖、噪声、语义递推和异常作用位置。
- `Domain Challenger`: 主动寻找不符合物理、统计或业务常识的机制，并给出必须修正的条件。
- `Plan Compiler`: 将前三者的证据编译成唯一、可执行的 `SeriesPlan`。
- `Reflection Agent`: 审查编译结果，必要时触发一次重规划。

系统固定运行完整多 Agent 工作流，不再提供单 Agent 或自动路由模式。多个角色可以共用同一个模型服务，但具有独立 Prompt、上下文和结构化输出。

规划优先级：

- 有 `OPENAI_API_KEY` 时使用完整 LangChain 多 Agent 工作流，Reflection 审查并按需触发一次重规划。
- 如果没有合适的已知工具: LLM 可以调用 `create_custom_domain_rule` 创建新领域规则，规则会写入 `domain_rules.json`。
- LangChain 多 Agent 调用失败、API key 缺失或额度不足时直接报错，不再自动回退到无 LLM 本地规则。
- 底层确定性生成器仍负责数值合成、语义变换、异常注入、约束修复和 Arrow 存储。

## 启动网页界面

安装依赖后运行：

```bash
conda activate generation-agent
cd /Users/chenzijie/Documents/Project/python/generation-agent
pip install -r requirements.txt
python -m generation_agent.web_app
```

浏览器打开 `http://127.0.0.1:7860`。页面支持：

- 单序列模式：填写描述、输出文件夹和 Arrow 文件名，生成一个紧凑 Arrow IPC 文件。
- 数据集模式：填写领域、输出文件夹和序列数量，生成多个独立 `.arrow` 分片、场景定义与清单文件。
- 公共参数：序列长度、频率、起始时间、随机种子、模型、参考 Arrow、异常开关和异常强度。
- 右侧会展示生成序列的抽样趋势图，只绘制少量等间距点以避免浏览器加载全量数据。

大规模数据集默认使用 Arrow IPC 紧凑存储：每行只保存 `value`、`anomaly` 和必要的语义数值列，`value` 为 `float32`；时间戳通过 `start`、`frequency`、`length` 在 manifest/Arrow metadata 中恢复，`unit/domain/generator_type/semantic_type` 等行级重复元数据也放入 manifest/Arrow metadata。

默认只监听本机。修改端口可以使用：

```bash
python -m generation_agent.web_app --port 7861
```

## 启动 MCP server

先安装依赖：

```bash
pip install -r requirements.txt
```

启动：

```bash
python -m generation_agent.mcp_server
```

MCP 工具说明：

- `plan_time_series(description, model=None)`: 生成结构化计划 JSON。
- `generate_time_series(..., anomalies="auto", anomaly_severity=None, ...)`: 直接生成时间序列，并允许覆盖异常开关和强度。
- `generate_time_series_code(...)`: 返回独立 Python 生成脚本。
- `synthesize_from_plan(plan_json, ...)`: 从计划 JSON 生成时间序列。

## 示例输出字段

- `timestamp`: 时间戳
- `value`: 生成值
- `anomaly`: 是否为异常点，`0/1`
- `unit`: 单位
- `domain`: 数据领域
- `generator_type`: 使用的领域生成机制
- `semantic_type`: 使用的输出语义机制

根据语义类型，输出还可能包含：

- `increment`: 累计量的每期增量。
- `inflow`、`outflow`、`net_flow`: 存量流量过程。
- `state`: 状态切换过程的状态编号。
- `step`: 随机游走步长。
- `impulse`: 衰减恢复过程的冲击。
- `growth_rate`: 饱和增长速率。
- `driver`、`lagged_driver`: 多变量滞后过程。

确定性校验结果与 Series Auditor 结果会保存在运行时 DataFrame 属性及数据集 `manifest.json` 中。
