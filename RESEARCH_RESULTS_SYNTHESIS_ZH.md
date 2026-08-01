# 临床 QA 不确定性方法适用条件：三阶段结果总览

更新时间：2026-08-01

## 研究主题

本项目研究：

> **临床 QA 中不同不确定性方法的适用条件，以及答案形式、模型规模和
> 语义聚类质量如何影响 Semantic Entropy、P(True) 与 hidden-state
> probes。**

项目不试图证明某一种 UQ 方法在所有任务上最优。核心问题是：在什么答案
形式、模型能力和语义等价判断条件下，各方法分别可靠；能否用单次生成的
hidden-state probe，以更低成本保留有用的不确定性信号。

截至 2026-07-28，三个阶段和预先限定的机制诊断均已完成。2026-08-01
导师讨论后，新增一项论文收尾范围：补齐三种重点 UQ 方法的 calibration、
selective prediction、正确性标签人工审核和零重校准 PubMedQA transfer；
不重新开放 prompt、模型规模、Probe 训练或聚类扩展。当前不依赖 Isambard
的 P(True)/Probe 部分已经完成；SE calibration 和正确性标签人工审核仍待
完成，因此下述 calibration 结论是明确标注的阶段性结果。

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
| Cluster count 对答案错误 | 0.7475 | 0.8368 | 十次采样加聚类 |
| P(True)-Probe 对答案错误 | 0.7452 | 0.8357 | 跨目标的次要比较 |
| 10-sample normalized NLL 对答案错误 | 0.7382 | 0.8466 | 十次采样 token baseline |

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

### 冻结 Probe 的跨数据集结果

两个 block-24/LT Probe 随后不经重新训练、重校准、阈值调整或特征选择，
直接应用到官方 PubMedQA PQA-L 500 题。最终 Appendix-C context v2 prompt
使用与 BioASQ 相同的 Gemma 3 12B checkpoint，并把官方 abstract 作为证据；
模型决策准确率为 362/500（72.4%），错误率为 27.6%。

先用 AUROC 比较三个在两个数据集上都按相同定义计算的
correctness-ranking score：

| 方法 | BioASQ test AUROC | PubMedQA v2 AUROC | 变化 |
| --- | ---: | ---: | ---: |
| Accuracy-Probe | 0.8058 | 0.5901 | -0.2157 |
| Blind P(True) | 0.7900 | 0.6490 | -0.1410 |
| P(True)-Probe | 0.7452 | **0.6839** | -0.0613 |

这里的 P(True)-Probe BioASQ 数字是它对答案错误的跨目标排名，不是其
0.9026 的原始 teacher-target fidelity。三种方法在 PubMedQA 上均下降，
但 P(True)-Probe 下降最小，并成为 v2 上最强的错误排序方法；相反，
直接以 BioASQ correctness 训练的 Accuracy-Probe 下降到 0.5901。这个结果
说明“源域目标更直接”不保证跨数据集迁移更强。

PubMedQA v2 同批 500 题的完整单答案方法对比如下：

| 方法 | Error AUROC | AP |
| --- | ---: | ---: |
| Frozen P(True)-Probe | **0.6839** | **0.4519** |
| Blind P(True) | 0.6490 | 0.4210 |
| Verbalized confidence | 0.6402 | 0.4164 |
| Frozen Accuracy-Probe | 0.5901 | 0.3916 |
| Normalized NLL | 0.5424 | 0.3075 |
| Mean token entropy | 0.5423 | 0.3080 |
| Sequence NLL | 0.5400 | 0.3025 |
| Max token entropy | 0.5372 | 0.2914 |

AP 必须结合 PubMedQA v2 的 27.6% 错误率理解，不能与 BioASQ AP 直接比较。
此外，P(True)-Probe 对其原始 frozen blind-P(True) teacher target 的 AUROC
只有 0.5899，远低于 BioASQ 的 0.9026；因此 0.6839 应解释为迁移后的
cross-target error-ranking usefulness，而不是原始内部映射得到保留。

这次 transfer 没有生成十个高温答案，也没有运行 NLI 聚类，所以没有
PubMedQA SE、cluster count 或 sample-disagreement 结果。它是低成本
单答案方法的外部压力测试，不替代 BioASQ 主结果，也不支持某个 Probe
已经获得数据集无关的泛化能力。BioASQ 表中的 normalized NLL 是十次采样
平均，而 PubMedQA 表中是单个主答案的 normalized NLL，因此也不把两者的
数值差解释为严格的跨数据集退化。

完整证据见：

- `PHASE2_PROBE_PLAN.md`
- `archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`
- `archehr_sebaseline/docs/phase2_probe_completion_statistics.md`
- `archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`

### 阶段性 calibration 与 selective prediction 结果（不含 SE）

三个现有单答案分数采用同一协议：在 384 个 BioASQ validation 样本上拟合
一维 logistic calibration，只在 384 个 test 样本上评估，目标统一为
`incorrect=1`。Accuracy-Probe 的 test Brier 为 0.1776、log loss 为
0.5386、10-bin ECE 为 0.0891；blind P(True) 分别为 0.2156、0.6174 和
0.1922。Accuracy-Probe minus blind P(True) 的配对 bootstrap 结果为：

- AUROC `+0.01584`，95% CI `[-0.03850,+0.07163]`，排序差异仍未确定；
- Brier `-0.03794`，95% CI `[-0.06050,-0.01541]`；
- log loss `-0.07884`，95% CI `[-0.13526,-0.02177]`；
- 0.5--1.0 coverage AURAC `-0.01294`，95% CI
  `[-0.02322,-0.00316]`。

因此，当前证据支持 Accuracy-Probe 在固定 calibration 协议下提供更好的
概率质量和 selective prediction，但仍不支持其 AUROC 显著高于 blind
P(True)。分题型结果还显示，统一 calibrator 在 list 上对三种方法都不能
超过该题型自身的 prevalence baseline，说明 pooled calibration 会掩盖
答案形式带来的巨大错误率偏移。

把 BioASQ 拟合的 calibration mapping 原样应用到 PubMedQA v2 后，blind
P(True)、Accuracy-Probe 和 P(True)-Probe 的 Brier skill 分别为
`-0.7198`、`-0.7979` 和 `-0.7572`。这说明 ranking signal 的有限迁移不等于
概率 calibration 能迁移；没有使用 PubMedQA 标签重拟合或修补映射。

完整协议和阶段性结果见
`archehr_sebaseline/docs/phase2_uq_calibration_completion.md`。SE 在 Isambard
恢复前保持待定，不由单答案结果外推。

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

## 收尾边界

- 不再训练新的 Probe 或融合模型；
- 不再扩展 1B factoid/list；
- 不再增加 270M/27B 模型点；
- 不再进行 PubMedQA prompt 调整或错误案例调参；
- 不把 Claude clustering 描述为部署方案；
- 当前只推进固定的 calibration/selective prediction、correctness audit 和
  zero-refit PubMedQA calibration transfer。

论文写作与收尾分析同步推进；完成这组固定评估后，将结果写入 Results、
Discussion 和 Limitations，不继续增加无关实验分支。
