# 实验一：信息来源消融

本目录独立实现“信息来源消融”实验，用于评估个人临床先验信息、6个月纵向PPG及两者表示层融合对房颤负荷增加预测的贡献。PPG固定采用7天聚合尺度，每位患者包含26个时间窗口。出院时与6个月随访时的Holter房颤负荷仅用于确定结局标签，不作为模型输入。

本目录是整理后的干净代码包，仅包含实验代码、配置、依赖说明和测试。虚拟环境、Python缓存、正式实验结果、示意性结果、图片及Word报告均未包含；运行后生成的这些目录已由`.gitignore`排除。

## 1. 实验比较内容

实验包含7种条件：

| 条件标识 | 输入或消融设置 | 目的 |
|---|---|---|
| `clinical_only` | 仅完整个人临床先验 | 评估临床先验的独立预测价值 |
| `ppg_only` | 仅26周纵向PPG | 评估纵向PPG的独立预测价值 |
| `fusion_full` | 完整临床先验与纵向PPG表示层融合 | 作为完整融合模型 |
| `fusion_drop_demographic_lifestyle` | 融合模型删除人口学及生活方式 | 评价该类先验信息的边际贡献 |
| `fusion_drop_af_history` | 融合模型删除房颤及既往病史 | 评价该类先验信息的边际贡献 |
| `fusion_drop_laboratory_echo_ecg` | 融合模型删除实验室、超声和心电信息 | 评价该类先验信息的边际贡献 |
| `fusion_drop_procedure_medication` | 融合模型删除手术及用药信息 | 评价该类先验信息的边际贡献 |

临床分支通过关系注意力编码四类先验变量，PPG分支通过时间图编码器处理26个连续窗口。完整模型将两个分支输出的患者级表示拼接后进行二分类，属于表示层融合。

## 2. 输入数据

代码以只读方式加载以下文件：

```text
../outputs/evoaf_synthetic_387/intermediate/
├── patient_baseline.csv   # 387例患者的个人临床先验变量
├── ppg_7day.csv           # 7天尺度聚合的纵向PPG特征，每例26个窗口
└── holter_outcome.csv     # 房颤负荷增加标签及预先确定的交叉验证折
```

`src/data.py`会检查患者数、临床字段及每例患者的PPG窗口数。数据预处理参数仅在各外层训练折内拟合，避免测试折信息泄漏。

## 3. 运行环境

推荐使用目录内已建立的虚拟环境。若需要重新安装依赖：

```powershell
cd C:\Users\20716\Desktop\恶化论文\gpt\experiment_1_clean_code
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

主要依赖包括PyTorch、pandas、NumPy、scikit-learn、Matplotlib和python-docx，具体版本见`requirements.txt`。

## 4. 完整运行流程

在本目录依次执行：

```powershell
.\.venv\Scripts\python.exe run_experiment.py
.\.venv\Scripts\python.exe make_results.py
.\.venv\Scripts\python.exe build_report.py
```

三个入口的职责不同：

1. `run_experiment.py`：训练7种模型条件，执行5折交叉验证和3个随机种子重复，保存逐患者折外预测、逐折训练信息和运行配置。
2. `make_results.py`：读取折外预测，计算总体性能、Bootstrap置信区间及相对完整融合模型的配对差值，同时生成ROC曲线和消融性能图。**该文件是实验一的结果统计与绘图Python文件。**
3. `build_report.py`：读取结果表和图片，生成Word格式的完整实验说明文档。

如已存在有效的`results/oof_predictions.csv`，仅需重新统计、绘图或生成报告时，可只运行后两条命令，无需重新训练模型。

## 5. 结果保存位置

模型运行后的最终说明文档保存在：

```text
report/实验一_信息来源消融_实验说明.docx
```

结构化结果保存在`results/`：

| 文件 | 内容 |
|---|---|
| `oof_predictions.csv` | 387例患者在7种条件下的折外预测概率、真实标签和验证集阈值；所有总体指标均可由此重新计算 |
| `summary_metrics.csv` | 各条件的AUROC、AUPRC、准确率、敏感度、特异度、F1分数及Bootstrap 95%置信区间 |
| `paired_differences.csv` | 各条件与完整融合模型之间的配对性能差值及置信区间 |
| `fold_metrics.csv` | 各条件、外层折和随机种子的验证AUROC、训练轮数、阈值及测试样本数 |
| `training_log.csv` | 每次训练任务的完成状态 |
| `cohort_summary.csv` | 实验队列规模及结局分布摘要 |
| `quality_checks.json` | 预测行数、条件数、概率范围和标签一致性等质量检查 |
| `run_config.json` | 本次实际运行使用的超参数快照 |

绘图结果保存在`figures/`：

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `roc_curves.png` | `make_results.py` | 主要模态模型与完整融合模型的ROC曲线 |
| `ablation_performance.png` | `make_results.py` | 完整融合模型及四类临床先验删除条件的性能比较 |

## 6. 代码结构

```text
experiment_1_information_source_ablation/
├── config.json             # 交叉验证、训练和Bootstrap参数
├── requirements.txt        # Python依赖及版本
├── run_experiment.py       # 模型训练主入口
├── make_results.py         # 指标汇总和图片渲染入口
├── build_report.py         # Word实验说明文档生成入口
├── src/
│   ├── data.py             # 数据读取、字段分组、折内预处理和序列构建
│   ├── models.py           # 临床关系注意力、PPG时间图编码器及融合分类器
│   ├── train.py            # 消融条件、训练、早停、验证阈值和折外预测
│   └── evaluate.py         # 性能指标、Bootstrap区间和配对差值
├── tests/                  # 数据、模型、训练、评价及输出测试
├── results/                # CSV和JSON结构化结果
├── figures/                # 论文和报告可用图片
└── report/                 # 最终Word实验说明文档
```

## 7. 当前实验参数

`config.json`当前设置为5折外层交叉验证、每折3个随机种子、训练集内20%验证集、最多160轮训练、早停耐心值20、隐藏维度32、2个注意力头和2000次分层Bootstrap。修改参数后应重新运行全部三步，以保证结果表、图片和报告一致。

## 8. 结果解释原则

- `clinical_only`、`ppg_only`与`fusion_full`用于判断两类信息的独立价值及表示层融合增益。
- 四个`fusion_drop_*`条件必须与`fusion_full`配对比较，用于评价某一类临床先验被删除后的性能变化。
- 主要结论应基于折外预测和置信区间，不应仅依据单一折或单一随机种子的结果。
- 本实验使用模拟的387例患者数据，因此当前结果用于验证实验流程和分析方法，不能直接作为真实队列的临床效能结论。

## 9. 示意性模拟结果

需要展示预期性能排序时，可运行独立的示意性模拟流程：

```powershell
.\.venv\Scripts\python.exe run_illustrative_simulation.py
```

该流程生成逐患者示意性预测，再调用同一个`make_results.py`计算指标、Bootstrap 95%置信区间并绘图。所有产物保存在`illustrative_simulation/`，不会覆盖正式实验的`results/`、`figures/`或`report/`。最终说明文档为：

```text
illustrative_simulation/report/实验一_信息来源消融_示意性模拟结果说明.docx
```

这些结果明确标注为示意性模拟结果，只能用于展示分析与制图形式。
