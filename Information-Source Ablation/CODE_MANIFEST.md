# 实验一代码包清单

## 正式实验入口

- `run_experiment.py`：训练7个实验条件并输出逐患者折外预测。
- `make_results.py`：计算指标、Bootstrap 95%置信区间、配对差值并绘图。
- `build_report.py`：生成正式实验说明Word文档。

## 核心实现

- `src/data.py`：读取数据、临床变量分组、折内预处理和PPG序列构建。
- `src/models.py`：临床关系注意力编码器、PPG时间图编码器和表示层融合分类器。
- `src/train.py`：实验条件、交叉验证、训练、早停、阈值和折外预测。
- `src/evaluate.py`：AUROC、AUPRC、敏感度、特异度、F1、Brier及Bootstrap计算。

## 配置与测试

- `config.json`：模型训练、交叉验证和Bootstrap参数。
- `requirements.txt`：固定版本的Python依赖。
- `tests/`：数据、模型、训练、指标、输出和示意流程测试。

## 示意性结果工具

- `illustrative.py`：生成明确标注的示意性患者级预测。
- `make_illustrative_predictions.py`：保存示意性折外预测。
- `run_illustrative_simulation.py`：运行示意性结果完整流程。
- `build_illustrative_report.py`：生成示意性结果说明文档。
- `build_confusion_audit_report.py`：重新计算混淆矩阵和分类指标并生成核对文档。

## 未包含内容

- `.venv/`及任何本地Python环境。
- `__pycache__/`、`.pytest_cache/`和Matplotlib缓存。
- `results/`、`figures/`、`report/`及渲染中间文件。
- `illustrative_simulation/`中的任何生成结果。
- 输入数据。正式实验默认从相邻的`../outputs/evoaf_synthetic_387/intermediate/`读取数据。
