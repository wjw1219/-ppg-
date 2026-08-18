# 实验二代码（干净版）

本目录只保存实验二“PPG观测窗口聚合比较”的代码、配置和测试，不包含数据、训练结果、图片、Word报告、虚拟环境或缓存文件。

实验二固定完整182天观测范围、个人先验信息、表示层融合模型、患者级五折划分和评价流程，仅比较3天、7天和14天PPG聚合窗口。三个尺度分别训练独立模型，7天为实验前预设主分析尺度。

## 目录结构

```text
experiment_2_window_aggregation_comparison_clean/
├── config.json
├── run_experiment.py
├── make_results.py
├── build_report.py
├── build_simulated_prediction_report.py
├── build_confusion_metric_audit_report.py
├── generate_simulated_oof_predictions.py
├── generate_weekly_signal_scenario.py
├── tune_simulated_prediction_parameters.py
├── src/
├── tests/
└── scenarios/weekly_signal_scenario/
```

## 输入数据

默认从本目录上一级项目目录读取：`../outputs/evoaf_synthetic_387/intermediate/`。需要 `patient_baseline.csv`、`ppg_3day.csv`、`ppg_7day.csv`、`ppg_14day.csv` 和 `holter_outcome.csv`。Holter房颤负荷数值只用于形成标签，不进入模型输入。

## 正式实验运行

```powershell
cd "C:\Users\20716\Desktop\恶化论文\gpt\experiment_2_window_aggregation_comparison_clean"
$python = "..\experiment_1_information_source_ablation\.venv\Scripts\python.exe"
& $python run_experiment.py
& $python make_results.py
& $python build_report.py
```

`run_experiment.py`训练3天、7天和14天三个独立模型并保存折外预测；`make_results.py`重新计算AUROC、AUPRC、敏感度、特异度、准确率、F1和Brier，执行2,000次患者级分层Bootstrap并绘图；`build_report.py`根据最新结果生成Word报告。

## 模拟预测与核对

`generate_simulated_oof_predictions.py`只生成明确标记为模拟的折外概率，不能替代真实模型训练结果。生成后仍必须运行 `make_results.py`，所有指标和置信区间都由统一评价代码重新计算。

`build_confusion_metric_audit_report.py`逐组核对TP、FN、TN、FP，并检查敏感度、特异度、F1、准确率和Brier是否与 `summary_metrics.csv` 一致。

`generate_weekly_signal_scenario.py`、`tune_simulated_prediction_parameters.py`及`scenarios/weekly_signal_scenario/`用于方法学模拟情景，不属于真实临床结果。

## 测试

```powershell
& $python -m unittest discover -s tests -v
```

代码测试可直接运行；依赖已有结果文件的输出结构测试，需要先运行正式实验。

## 依赖

Python 3.12环境，主要依赖 `numpy`、`pandas`、`scikit-learn`、`scipy`、`torch`、`matplotlib` 和 `python-docx`。可复用 `../experiment_1_information_source_ablation/.venv/`，本目录不复制虚拟环境。
