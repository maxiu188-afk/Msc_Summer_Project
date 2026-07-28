# 临床 QA 不确定性方法适用条件：三阶段结果总览

更新时间：2026-07-28

## 研究主题

本项目研究：

> **临床 QA 中不同不确定性方法的适用条件，以及答案形式、模型规模和
> 语义聚类质量如何影响 Semantic Entropy、P(True) 与 hidden-state
> probes。**

项目不试图证明某一种 UQ 方法在所有任务上最优。核心问题是：在什么答案
形式、模型能力和语义等价判断条件下，各方法分别可靠；能否用单次生成的
hidden-state probe，以更低成本保留有用的不确定性信号。

截至 2026-07-28，三个阶段和预先限定的机制诊断均已完成。与导师讨论前不再
新增实验、调整 prompt、扩展模型规模或训练新的 Probe。

## 四个连续结论

### 1. UQ 方法的相对表现依赖答案形式

Factoid、list 和 summary 不能只用一个总体指标概括。Gemma 3 12B 的两次
Phase-1 运行重复得到：

- factoid 上 blind P(True) 比 SE 高约 0.075 AUROC；
- list 上 SE 与 blind P(True) 基本相当，cluster count 是两个 seed 中最强的
  list 点估计；
- summary 上 blind P(True) 比 SE 高约 0.23--0.24 AUROC。

因此，短实体答案、集合答案和长自由文本答案对应不同的 UQ operating
regime。后续配对缩短实验没有恢复 summary SE，说明观察到的差异不能简化为
“答案越长，SE 越差”；答案结构和语义聚类难度同样重要。

### 2. P(True) 更依赖模型自身能力

在同一批对齐的 summary 问题上，blind P(True) AUROC 从 12B 的 0.807
下降到 4B 的 0.652，再下降到 1B 的 0.501。P(True)-minus-SE 的点估计依次
为 `+0.243`、`+0.008` 和 `-0.096`。

12B 到 4B 的相对变化已经由配对 bootstrap 支持；1B 相对 4B 的增量变化
区间仍跨零。因此可以说 P(True) 的自我评估优势随模型能力下降而消失，
但不能声称 1B 与 4B 已被新的显著性结果完全区分。

### 3. SE 对模型规模相对稳定，但依赖语义聚类质量

同一 summary 对齐队列上，SE AUROC 在 12B、4B、1B 分别约为
0.564、0.644、0.597，波动明显小于 P(True)。这不是 SE 在小模型上变强的
证据：1B/4B 上 normalized NLL 仍优于 SE，且答案正确率已经显著下降。

更直接的机制证据来自固定生成答案的聚类替换：

- 原 PubMedBERT NLI 聚类下，93 题 SE AUROC 为 0.595；
- Claude 聚类下，SE AUROC 为 0.781；
- 改变量为 `+0.187`，95% CI `[+0.080,+0.291]`；
- 原 P(True)-minus-SE gap 的点估计约有 85% 被关闭。

24 对盲审进一步显示，Claude 聚类与人工一致 19/24，而原 NLI 只有 9/24。
原 NLI 同时出现错误合并和错误拆分；Claude 没有错误合并，剩余错误主要是
偏保守的过度拆分。由于这 24 对是按 NLI/Claude 组合分层抽取的诊断样本，
79.2% 不能解释为完整数据上的总体准确率。

### 4. 单次生成 Probe 是有竞争力的低成本替代方案

Accuracy-Probe 在冻结 BioASQ test 上达到 0.8058 AUROC / 0.8884 AP，
点估计高于 blind P(True) 的 0.7900 / 0.8383，也高于需要十次采样和 NLI
聚类的 SE。两者 AUROC 差异的 95% CI 为 `[-0.0389,0.0717]`，所以不能声称
Accuracy-Probe 已显著超过 blind P(True)。

两类 Probe 回答不同问题：

- P(True)-Probe 以 direct blind P(True) 为监督目标，主要衡量内部
  self-evaluation 信号的低成本复现；其目标内 held-out AUROC 为 0.9026。
- Accuracy-Probe 直接以答案正确/错误为监督目标，主要衡量 correctness
  ranking；其 held-out AUROC 为 0.8058。

这种目标差异比构造一个统一 Probe 更重要。预测模型自身不确定性和直接预测
答案错误不是同一个任务，在数据或 prompt 变化时也不一定以同样方式迁移。

## Phase 1：建立不同答案形式下的 UQ 基准

### 目标

在统一的 no-evidence BioASQ 协议下，比较 SE、blind P(True)、NLL/token
baselines 和简单分歧指标，并确定不同题型是否对应不同适用条件。

### 设计

- 模型：Gemma 3 12B；
- 1,000 个固定问题：480 factoid、320 list、200 summary；
- 两个生成 seed：31、47；
- 每题十个 `T=1.0` 样本用于 SE，一个 `T=0.1` 主答案用于正确性判断；
- 两个 seed 各有 991 个有效 Claude binary correctness labels；
- `incorrect=1`，跨题型比较以 AUROC 为主。

### 主结果

| 题型 | Mean sampled tokens | SE AUROC | Blind P(True) AUROC | P(True) minus SE |
| --- | ---: | ---: | ---: | ---: |
| Factoid | 8.08 / 8.01 | 0.720 / 0.725 | 0.795 / 0.800 | +0.075 / +0.075 |
| List | 35.42 / 35.41 | 0.845 / 0.881 | 0.845 / 0.852 | +0.000 / -0.029 |
| Summary | 111.13 / 110.92 | 0.565 / 0.595 | 0.808 / 0.826 | +0.243 / +0.231 |

Phase 1 的主结论不是“P(True) 全面优于 SE”，而是相对表现具有明确的
题型依赖：SE 对 list 很有竞争力，但当前 whole-answer NLI 聚类下不适合长
summary；P(True) 在 factoid 和 12B summary 上更强。

完整证据见
`archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md`。

## Phase 2：训练和评估单次生成 hidden-state probes

### 目标

测试一个简单线性 Probe 能否：

1. 低成本复现 direct P(True) 的内部不确定性信号；
2. 直接预测主答案是否错误。

### 设计

- 3,930 个 BioASQ 问题；
- train/validation/test = 3,144/393/393；
- Phase-1 的 1,000 个已观察问题只进入 train；
- layer/token 仅在 validation 选择；
- 最终 block 24/LT 配置只在 test 评估一次；
- Accuracy-Probe 使用 384 个有效 test correctness labels。

### 主结果

| 方法或目标 | AUROC | AP | 解释 |
| --- | ---: | ---: | --- |
| P(True)-Probe 对自身 P(True) target | 0.9026 | 0.8948 | 目标内 fidelity |
| Accuracy-Probe 对答案错误 | 0.8058 | 0.8884 | 直接 correctness ranking |
| Blind P(True) 对答案错误 | 0.7900 | 0.8383 | 二次 self-evaluation |
| Discrete SE 对答案错误 | 0.7484 | 0.8446 | 十次采样加聚类 |
| P(True)-Probe 对答案错误 | 0.7452 | 0.8357 | 跨目标的次要比较 |

Accuracy-Probe minus blind P(True) 为 `+0.01584` AUROC，配对 bootstrap
95% CI `[-0.03890,0.07173]`。Probe 的主要优势因此是竞争力与成本，而不是
已经证实的显著精度领先。

### 成本

| 方法 | Mean latency/question | 相对说明 |
| --- | ---: | --- |
| Accuracy-Probe | 60.40 ms | 单次 hidden-state replay |
| P(True)-Probe | 60.50 ms | 单次 hidden-state replay |
| 两个 Probe 共享 replay | 60.54 ms | 两个线性 head 的额外成本可忽略 |
| Blind P(True) | 154.37 ms | 两个固定 continuation score |
| 10-sample normalized NLL | 30.55 s | 十次自由生成 |
| SE | 32.75 s | 十次生成加 NLI 聚类 |

Probe 约比 blind P(True) 快 2.55 倍，比采样方法快 500 倍以上。

冻结 Probe 在 PubMedQA Appendix-C context v2 上作为有限的跨数据集检查：
P(True)-Probe error AUROC 为 0.6839，Accuracy-Probe 为 0.5901。该结果支持
“监督目标会影响迁移方式”，但 PubMedQA 不替代 BioASQ 主结果。

完整证据见：

- `PHASE2_PROBE_PLAN.md`
- `archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`
- `archehr_sebaseline/docs/phase2_probe_completion_statistics.md`
- `archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`

## Phase 3：解释答案形式、模型规模和聚类质量的影响

### 3A. 答案长度不是充分解释

在 121 个 paired summary 问题上，短 prompt 将主答案从 82.88 个词降到
52.11 个词，同时准确率保持 56/121。P(True)-minus-SE 的相对变化为
`+0.0405`，95% CI `[-0.0713,+0.1531]`。

结论：控制性缩短没有恢复 SE，长 summary 的问题不能只归因于字数。

### 3B. P(True) 的优势随模型能力下降

正式三模型比较使用 196 个共同有效的 summary 问题：

| 模型 | Accuracy | Blind P(True) AUROC | SE AUROC | Normalized NLL AUROC | P(True) minus SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 1B | 0.122 | 0.501 | 0.597 | 0.789 | -0.096 |
| Gemma 3 4B | 0.281 | 0.652 | 0.644 | 0.796 | +0.008 |
| Gemma 3 12B | 0.459 | 0.807 | 0.564 | 0.756 | +0.243 |

1B-minus-4B gap change 为 `-0.104 [-0.267,+0.059]`；1B-minus-12B 为
`-0.338 [-0.508,-0.157]`。模型能力主要影响 P(True) 的自我评估可靠性，
而不是证明 SE 在小模型上普遍更强。

1B 的 50-question feasibility gates 只有 8/50 factoid 和 3/50 list 正确，
因此没有扩展其余题型。

### 3C. 长答案 SE 对语义聚类质量敏感

固定相同生成答案，只替换语义等价判断：

| 方法 | 93-question SE AUROC | 24-pair human agreement | 错误合并 | 错误拆分 |
| --- | ---: | ---: | ---: | ---: |
| 原 PubMedBERT NLI clustering | 0.595 | 9/24 | 5 | 10 |
| Claude clustering | 0.781 | 19/24 | 0 | 5 |
| Claude direct pair judgement | — | 17/24 | 0 | 7 |

人工审核为按 NLI/Claude 判断组合分层抽取的机制诊断，不能估计全体 answer
pairs 的总体准确率。它支持的边界结论是：

> 原 NLI 是长回答 SE 下降的重要误差来源。Claude 替换显著减少错误合并，
> 且整体更符合人工语义判断，但仍存在偏保守的过度拆分。

Claude 的 API 成本、延迟和直接判断的非传递性使其不适合作为当前部署方案。
本阶段诊断的是瓶颈，不是提出新的生产聚类器。

完整证据见：

- `archehr_sebaseline/docs/summary_length_intervention.md`
- `archehr_sebaseline/docs/summary_clustering_diagnostic.md`
- `archehr_sebaseline/docs/gemma3_model_scale_experiment.md`

## 主线与归档边界

主线只保留：

1. Phase-1 分题型 UQ 基准；
2. Phase-2 两个冻结 Probe、held-out 结果、成本与有限 transfer；
3. Phase-3 长度控制、模型规模和语义聚类质量诊断。

以下内容保留用于 provenance，但不再承担论文主结论：

- 早期 100-question pilots 和旧 evidence-conditioned summary runs；
- Phase-1 post-hoc feature fusion；
- Phase-2 validation-fitted two-Probe fusion；
- question-only PubMedQA transfer；
- 已完成会议的旧版本简报和过时运行说明。

归档入口：`archive_unused/README.md`。

## 最终边界

- 不再训练新的 Probe 或融合模型；
- 不再扩展 1B factoid/list；
- 不再增加 270M/27B 模型点；
- 不再进行 PubMedQA prompt 调整或错误案例调参；
- 不把 Claude clustering 描述为部署方案；
- 与导师讨论前不再新增实验。

下一阶段是将三阶段结果写入论文的 Results、Discussion 和 Limitations，
而不是继续增加实验分支。
