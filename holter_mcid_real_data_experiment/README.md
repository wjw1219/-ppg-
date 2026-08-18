# 两次Holter房颤负荷研究特异性MCID分析

本目录是两次Holter房颤负荷研究特异性MCID分析的独立代码包。

- 主程序：`holter_mcid_real_data_analysis.py`
- 一键运行：`run.ps1`
- 依赖列表：`requirements.txt`

该程序用于真实患者数据，不会生成模拟临床锚定、模拟事件或模拟模型预测。分析的目标是：在真实两次Holter数据和独立临床锚定数据的基础上，数据驱动地估计本研究的研究特异性暂定MCID。

## 推荐数据结构

最简单的方式是准备一个合并后的患者级CSV，每位患者一行：

```text
patient_id,holter1_af_burden_pct,holter2_af_burden_pct,anchor_positive,stable_subgroup
P001,12.4,23.1,1,0
P002,8.2,7.4,0,1
```

必需字段：

- `patient_id`
- 两次Holter房颤负荷，取值范围为0–100，单位为百分比。程序也会自动识别 `discharge_af_burden_pct` 和 `month6_af_burden_pct`。
- 独立临床锚定：推荐使用二分类的 `anchor_positive`，编码为0/1。

用于MDC95估计的稳定性信息至少需要以下一种：

- `stable_subgroup`：稳定患者标记，编码为0/1。程序将在稳定患者中使用两次Holter差值估计重复测量误差；或
- 单独的重复测量差值列，通过 `--stable-repeat-diff-column` 指定，单位为pp。

如果只有两次临床随访Holter、没有稳定患者或重复测量数据，程序会明确输出 `MDC95 not estimable`，不会自行填充MDC95。

## 临床锚定评分

如果没有二分类锚定，但有连续的真实临床锚定评分，可以使用：

```powershell
--anchor-score-column clinical_anchor_score
```

程序会将实际评分最高的三分之一定义为锚定阳性。这个标签来自真实临床锚定评分，不是根据Holter差值生成的。

## 分开存放Holter和临床数据

也可以将文件分开：

`data/paired_holter.csv`

```text
patient_id,discharge_af_burden_pct,month6_af_burden_pct,stable_subgroup
```

`data/clinical_anchor.csv`

```text
patient_id,anchor_positive
```

`data/cohort.csv`可选，用于报告入组总人数：

```text
patient_id
P001
P002
```

## 运行命令

如果数据放在当前目录的`data`文件夹中，推荐运行：

在工作区根目录执行：

```powershell
& ".\holter_mcid_real_data_experiment\run.ps1"
```

如果Windows执行策略阻止脚本运行，可使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\holter_mcid_real_data_experiment\run.ps1"
```

也可以直接指定路径：

```powershell
& ".\holter_mcid_real_data_experiment\run.ps1" `
  -Holter "data\paired_holter.csv" `
  -Clinical "data\clinical_anchor.csv" `
  -Cohort "data\cohort.csv" `
  -OutputDir "results\holter_mcid_real"
```

如果已经合并为一个文件：

```powershell
& ".\experiment_1_information_source_ablation\.venv\Scripts\python.exe" `
  holter_mcid_real_data_experiment\holter_mcid_real_data_analysis.py `
  --input "data\patient_holter_mcid.csv" `
  --output-dir "results\holter_mcid_real" `
  --anchor-positive-column "anchor_positive" `
  --stable-column "stable_subgroup" `
  --bootstrap 2000 `
  --seed 20260818
```

如果实际列名不同，使用显式映射：

```powershell
--holter-1-column "Holter_Baseline" `
--holter-2-column "Holter_6m" `
--anchor-positive-column "Clinical_Worsening" `
--stable-column "Stable_Patient"
```

## 分析规则

1. 房颤负荷变化定义为：

   ```text
   delta_burden_pp = second Holter AF burden - first Holter AF burden
   ```

   这里的pp是percentage points，不是相对百分比变化。

2. 在0–20 pp范围内，以0.5 pp为步长搜索候选阈值。

3. 以Youden指数最大值选择阈值。若出现完全相同的Youden指数，选择较低阈值。

4. 进行2000次患者层面bootstrap，评估最佳阈值的中位数和95%重抽样范围。

5. MDC95使用稳定患者的重复差异计算：

   ```text
   MDC95 = 1.96 × SD(stable repeat differences)
   ```

6. `delta >= 10 pp`、`-10 < delta < 10 pp`和`delta <= -10 pp`三组只用于描述性分层，不参与阈值生成。

## 输出文件

- `patient_level_analysis.csv`：纳入分析的患者级数据和QC状态。
- `quality_control_log.csv`：每位患者的纳入或排除原因。
- `threshold_grid.csv`：0–20 pp所有候选阈值的性能指标。
- `selected_threshold.csv`：Youden指数最大对应的阈值。
- `bootstrap_selected_thresholds.csv`：2000次患者级bootstrap的最佳阈值。
- `bootstrap_threshold_summary.csv`：bootstrap中位数和95%范围。
- `mdc95_summary.csv`：MDC95及其计算来源。
- `anchor_summary_at_selected_cutoff.csv`：最佳阈值上下方的锚定阳性率。
- `change_group_counts_at_10pp.csv`：10 pp描述性分组数量。
- `figure_1_cohort_and_holter_change.svg/pdf`：队列、配对Holter和连续差值分布。
- `figure_2_threshold_discovery.svg/pdf`：Youden、敏感度/特异度和锚定阳性率。
- `figure_3_clinical_anchor.svg/pdf`：bootstrap稳定性、MDC95和锚定概率曲线。
- `figure_captions_latex.tex`：三张图的LaTeX图注。
- `analysis_metadata.json`和`analysis_log.txt`：分析配置、数据来源和运行记录。

## 重要解释边界

该流程得到的是本研究队列和临床锚定定义下的“研究特异性暂定MCID”。即使数据驱动结果接近10 pp，也不能直接声称10 pp是房颤负荷领域普适性的MCID。正文中应同时报告真实临床锚定定义、样本量、bootstrap稳定性和MDC95来源。
