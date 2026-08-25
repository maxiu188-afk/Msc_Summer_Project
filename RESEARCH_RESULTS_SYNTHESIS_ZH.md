# 临床 QA 不确定性方法适用条件：三阶段结果总览

更新时间：2026-08-07

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
不重新开放 prompt、模型规模、Probe 训练或聚类扩展。完整 calibration、
selective prediction、PubMedQA transfer 和正确性标签人工审核现已全部完成。
之后单独批准的 PubMedQA SE 温度敏感性诊断也已完成；在固定 200 题和
`T=0.7–1.3` 范围内，未发现 SE 排序表现受到明显影响。论文收尾实验到此冻结。

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
证据：1B/4B 上 `10-sample normalized NLL` 仍优于 SE，且答案正确率已经显著
下降。

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

## NLL 命名约定

本报告中的 NLL 只有两种基础公式，但由于单答案与十次采样不是同一个 UQ
估计量，统一使用以下四个名称：

| 统一名称 | 生成协议 | 题目级计算 | 历史别名或字段 |
| --- | --- | --- | --- |
| `10-sample normalized NLL` | 十个 `T=1.0` 样本 | 每个样本先计算 per-token NLL，再对十个样本取平均 | `predictive_entropy`、`mean_normalized_nll`、negative mean token log-probability、`avg_token_logprob_uncertainty` |
| `10-sample sequence NLL` | 十个 `T=1.0` 样本 | 每个样本先对全部生成 token 的 NLL 求和，再对十个样本取平均 | mean sequence NLL、G-NLL |
| `single-answer normalized NLL` | 一个 `T=0.1` 主答案 | 主答案的 per-token NLL | 单答案产物中的 `mean_normalized_nll` |
| `single-answer sequence NLL` | 一个 `T=0.1` 主答案 | 主答案全部生成 token 的 NLL 总和 | 单答案产物中的 `mean_sequence_nll` |

因此，历史字段名 `predictive_entropy` 在当前实现中不作为独立方法报告；它与
`10-sample normalized NLL` 数值和排序相同。Normalized 与 sequence NLL 仍是
不同方法，后者没有长度归一化，必须保留答案长度混杂这一解释边界。字段名前的
`mean_` 也不能单独判断采样协议：十样本产物表示跨十个答案平均，单答案产物中
则只是一个答案。

## Phase 1：建立不同答案形式下的 UQ 基准

### 目标

在统一的 no-evidence BioASQ 协议下，比较 SE、blind P(True)、NLL/token
baselines 和简单分歧指标，并确定不同题型是否对应不同适用条件。

### 设计

- 模型：Gemma 3 12B；
- 1,000 个固定问题：480 factoid、320 list、200 summary；
- 两个生成 seed：31、47；
- 每题十个 `T=1.0` 样本用于 SE、`10-sample NLL` 和 token baselines，一个
  `T=0.1` 主答案用于正确性判断；
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
| `10-sample normalized NLL` 对答案错误 | 0.7382 | 0.8466 | 十次采样 token baseline |

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
| `10-sample normalized NLL` | 30.55 s | 十次自由生成 |
| SE | 32.75 s | 十次生成加 NLI 聚类 |

Probe 约比 blind P(True) 快 2.55 倍，比采样方法快 500 倍以上。

### 冻结 Probe 的跨数据集结果

两个 block-24/LT Probe 随后不经重新训练、重校准、阈值调整或特征选择，
直接应用到官方 PubMedQA PQA-L 500 题。最终 Appendix-C context v2 prompt
使用与 BioASQ 相同的 Gemma 3 12B checkpoint，并把官方 abstract 作为证据；
模型决策准确率为 362/500（72.4%），错误率为 27.6%。

先用 AUROC 比较四个在两个数据集上都按相同定义计算的
correctness-ranking score：

| 方法 | BioASQ test AUROC | PubMedQA v2 AUROC | 变化 |
| --- | ---: | ---: | ---: |
| Accuracy-Probe | 0.8058 | 0.5901 | -0.2157 |
| Blind P(True) | 0.7900 | 0.6490 | -0.1410 |
| P(True)-Probe | 0.7452 | **0.6839** | -0.0613 |
| Discrete Semantic Entropy | 0.7484 | 0.5884 | -0.1600 |

这里的 P(True)-Probe BioASQ 数字是它对答案错误的跨目标排名，不是其
0.9026 的原始 teacher-target fidelity。四种方法在 PubMedQA 上均下降，
但 P(True)-Probe 下降最小，并成为 v2 上最强的错误排序方法；相反，
直接以 BioASQ correctness 训练的 Accuracy-Probe 下降到 0.5901。这个结果
说明“源域目标更直接”不保证跨数据集迁移更强。

PubMedQA v2 同批 500 题的完整方法对比如下；除明确标注的 SE 外，其余均为
原冻结 transfer 的单答案方法：

| 方法 | Error AUROC | AP |
| --- | ---: | ---: |
| Frozen P(True)-Probe | **0.6839** | **0.4519** |
| Blind P(True) | 0.6490 | 0.4210 |
| Verbalized confidence | 0.6402 | 0.4164 |
| Frozen Accuracy-Probe | 0.5901 | 0.3916 |
| Discrete Semantic Entropy（十样本补充） | 0.5884 | 0.3674 |
| `single-answer normalized NLL` | 0.5424 | 0.3075 |
| Mean token entropy | 0.5423 | 0.3080 |
| `single-answer sequence NLL` | 0.5400 | 0.3025 |
| Max token entropy | 0.5372 | 0.2914 |

AP 必须结合 PubMedQA v2 的 27.6% 错误率理解，不能与 BioASQ AP 直接比较。
此外，P(True)-Probe 对其原始 frozen blind-P(True) teacher target 的 AUROC
只有 0.5899，远低于 BioASQ 的 0.9026；因此 0.6839 应解释为迁移后的
cross-target error-ranking usefulness，而不是原始内部映射得到保留。

原冻结 Probe transfer 本身没有生成十个高温答案，也没有运行 NLI 聚类；后续
单独批准的补充实验保持 v2 prompt、500 题、Gemma 3 12B、seed 31、错误标签和
PubMedBERT 双向 NLI 不变，只增加每题十个 `T=1.0` 样本。正式任务 5921809
完成 500 题和 5,000 次生成，得到 discrete SE AUROC 0.5884、AP 0.3674；AP
需与 0.276 的错误率基线一起理解，且未做新增显著性检验。

444/500 题的十个答案被合并为单一语义簇，另有 46/7/3 题形成 2/3/4 个簇。
这解释了 SE 分数变化较少以及整体排序偏弱，但不能只凭本次结果区分模型回答
确实一致和 PubMedBERT 过度合并。该补充不改变冻结 Probe transfer 的单答案
协议，也不支持数据集无关泛化。BioASQ 表中使用 `10-sample normalized NLL`，
而 PubMedQA 表中使用 `single-answer normalized NLL`，因此也不把两者的数值差
解释为严格的跨数据集退化。

为检验采样温度是否解释 SE 表现，最后在同一个确定性分层 200 题子集上比较
`T=0.7/1.0/1.3`，只生成缺少的 `T=0.7/1.3` 两臂并复用已验收的 `T=1.0`：

| 温度 | Error AUROC | AP | Mean discrete SE | 单簇比例 |
| ---: | ---: | ---: | ---: | ---: |
| 0.7 | 0.5837 | 0.3657 | 0.0514 | 90.5% |
| 1.0 | **0.6213** | 0.3810 | 0.0640 | 85.5% |
| 1.3 | 0.5945 | **0.3850** | 0.0550 | 88.5% |

相对 `T=1.0`，`T=0.7` 与 `T=1.3` 的 AUROC 变化分别为
`-0.0376 [-0.0913,+0.0123]` 和 `-0.0268 [-0.0774,+0.0213]`；AP 和平均
SE 的 20,000 次配对 bootstrap 区间同样跨零。因此在本次固定队列、单一
seed、模型和 NLI 规则下，未发现 `T=0.7–1.3` 对 SE 错误排序表现有明显影响；
`T=1.0` 只是 AUROC 点估计最高，不能解释为已确定的最优温度。

唯一有区间支持的结构变化是 `T=0.7` 让单簇比例增加 5 个百分点，95% CI
`[+1.0,+9.5]`。这说明低温会增加语义聚类塌缩，但没有转化为明确的 AUROC/AP
变化；`T=1.3` 也没有可靠增加语义多样性。结论不能外推为“温度永远不影响
SE”，而应限定为本次测试范围内没有检测到明显的 SE 表现变化。

完整证据见：

- `PROJECT_CLOSEOUT.md`
- `archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`
- `archehr_sebaseline/docs/phase2_probe_completion_statistics.md`
- `archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`

### Calibration 与 selective prediction 结果

四个分数采用同一协议：在 384 个 BioASQ validation 样本上拟合一维
logistic calibration，只在 384 个 test 样本上评估，目标统一为
`incorrect=1`。

| 方法 | AUROC | Brier | Brier skill | Log loss | ECE | AURAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy-Probe | **0.8058** | **0.1776** | **0.2125** | 0.5386 | 0.0891 | **0.2780** |
| Blind P(True) | 0.7900 | 0.2156 | 0.0444 | 0.6174 | 0.1922 | 0.2909 |
| Semantic Entropy | 0.7484 | 0.1839 | 0.1847 | **0.5341** | **0.0298** | 0.2799 |
| P(True)-Probe | 0.7452 | 0.2001 | 0.1131 | 0.5860 | 0.0984 | 0.2905 |

20,000 次配对 bootstrap 得到：

- AUROC `+0.01584`，95% CI `[-0.03850,+0.07163]`，排序差异仍未确定；
- Brier `-0.03794`，95% CI `[-0.06050,-0.01541]`；
- log loss `-0.07884`，95% CI `[-0.13526,-0.02177]`；
- 0.5--1.0 coverage AURAC `-0.01294`，95% CI
  `[-0.02322,-0.00316]`。

以上四项是 Accuracy-Probe minus blind P(True)。Accuracy-Probe 相对 SE 的
AUROC 优势为 `+0.05745 [+0.00528,+0.10949]`，但 Brier、log loss 和 AURAC
差异均跨零。SE 相对 blind P(True) 则在 Brier、log loss 和 AURAC 上均有
区间支持的优势。因此，ranking、calibration 和 selective prediction 不会
产生完全相同的方法排序。

随后批准的离线 operating-point 补充没有在 test 内重新取固定比例，而是在
384 个 validation 样本上冻结 `0.80/0.90/0.95` coverage 阈值，再原样应用到
384 个 test 样本。`0.80` 目标下，SE、blind P(True)、Accuracy-Probe 和
P(True)-Probe 的实际 coverage 分别为 `0.807/0.789/0.763/0.794`，retained
risk 为 `0.581/0.601/0.567/0.590`，均低于全覆盖风险 `0.656`。由于实际
coverage 不同，这些风险不能直接解释为方法间显著优胜。SE 的 `0.95` 阈值
因离散分数边界同分而保留全部 test 样本，说明它在高 coverage 运行点存在
阈值粒度限制；同一个全局 `0.80` 阈值还让 list coverage 仅为
`0.417–0.650`，而 summary 为 `0.951–1.000`。这是一项部署覆盖差异诊断，
没有拟合题型专属阈值。

分题型 Brier skill 进一步显示：SE 在 list 上为 `+0.1666`，是唯一超过该
题型 prevalence baseline 的方法；Accuracy-Probe 在 factoid 和 summary 上
分别为 `+0.1553` 和 `+0.1613`，表现最好。这把“方法表现依赖答案形式”的
主结论从 ranking 扩展到了 calibration。

把 BioASQ 拟合的 calibration mapping 原样应用到 PubMedQA v2 后，blind
P(True)、Accuracy-Probe 和 P(True)-Probe 的 Brier skill 分别为
`-0.7198`、`-0.7979` 和 `-0.7572`。这说明 ranking signal 的有限迁移不等于
概率 calibration 能迁移；没有使用 PubMedQA 标签重拟合或修补映射。

完整协议和结果见
`archehr_sebaseline/docs/phase2_uq_calibration_completion.md`。

### Correctness label 人工审核

固定审核样本为 BioASQ test 中按题型均衡抽取的 90 题，每类 30 题。审核者
只看到问题、相应参考答案和模型回答，不看到 Claude label、example ID 或
UQ score。90 题全部给出 `correct`/`incorrect` 判断，没有 `unsure`。

| 题型 | 一致率 | Cohen's kappa | Claude 判错、人工判对 | Claude 判对、人工判错 |
| --- | ---: | ---: | ---: | ---: |
| Overall | **95.56%** | **0.902** | 4 | 0 |
| Factoid | 93.33% | 0.857 | 2 | 0 |
| List | **100.00%** | **1.000** | 0 | 0 |
| Summary | 93.33% | 0.867 | 2 | 0 |

按完整 test 题型比例加权后的总体一致率为 `95.12%`。四个分歧全部是 Claude
判错而人工判对，涉及参考答案本身可能不理想、概括但核心正确的 summary，
以及额外但合法的信息。这说明 Claude correctness judge 整体可靠，但存在
轻微的保守倾向，可能略高估错误率；四个分歧不足以精确估计总体偏差。

在相同 90 题上，用原有 validation-fitted probability 做设计加权敏感性
诊断，不重拟合、不选模型。人工标签下 Accuracy-Probe、blind P(True)、SE
的 AUROC 分别为 `0.8939`、`0.8701`、`0.7919`，主方法排序未改变；对应
Brier 为 `0.1657`、`0.2240`、`0.1754`。因此四处标签修正没有推翻完整 test
上的 ranking 和 calibration 主结论。由于该样本按题型均衡抽取且只有 90
题，这些数值只承担标签敏感性诊断，不替代正式 test 结果，也不增加显著性
声明。

## Phase 3：解释答案形式、模型规模和聚类质量的影响

### 3A. 答案长度不是充分解释

在 121 个 paired summary 问题上，短 prompt 将主答案从 82.88 个词降到
52.11 个词，同时准确率保持 56/121。P(True)-minus-SE 的相对变化为
`+0.0405`，95% CI `[-0.0713,+0.1531]`。

结论：控制性缩短没有恢复 SE，长 summary 的问题不能只归因于字数。

### 3B. P(True) 的优势随模型能力下降

正式三模型比较使用 196 个共同有效的 summary 问题：

| 模型 | Accuracy | Blind P(True) AUROC | SE AUROC | `10-sample normalized NLL` AUROC | P(True) minus SE |
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
- 固定的 calibration/selective prediction、correctness audit 和 zero-refit
  PubMedQA calibration transfer 均已完成。
- validation-fixed selective-prediction operating-point 补充已完成；不再追加
  阈值调参或题型专属策略。

下一阶段是把现有证据写入 Results、Discussion 和 Limitations，不继续增加
无关实验分支。除非导师明确开启一个独立的新想法，否则实验保持冻结。
