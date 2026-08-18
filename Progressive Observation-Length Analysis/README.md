# 实验三：逐周减量与完整周数模型比较

这是实验三的干净代码目录，只包含可执行代码、配置、测试和说明，不包含任何已生成的CSV、PNG、DOCX、模型缓存或运行日志。运行脚本后，结果会在本目录下新建`results`、`figures`和`report`目录。

主训练入口是`run_experiment.py`；模型、数据预处理和训练逻辑分别位于`src/models.py`、`src/data.py`和`src/train.py`。`make_results.py`负责统计与作图，`benchmark_models.py`负责计算资源测量，`build_report.py`负责生成主实验说明DOCX。`expected_scenario`目录中的脚本用于生成明确标注为“预期结果模拟”的辅助数据和核对文档，不属于真实模型训练结果。

本目录独立保存实验三的训练、统计、资源测量、报告生成和测试代码，不与实验一、实验二混用。

实验包含两个部分：完整模型分别输入前1至前26周PPG数据，每个周数单独训练；在完整26周输入下比较特征拼接+MLP、GRU、LSTM、Transformer、普通时序GAT和完整模型。所有模型预测同一个结局，即6个月随访时72小时Holter房颤负荷是否较出院时增加。Holter房颤负荷只用于确定标签，不进入模型输入。

## Python文件调用关系

```text
patient_baseline.csv + ppg_7day.csv + holter_outcome.csv
                         |
                         v
                 src/data.py
                         |
                         v
src/models.py <--- src/train.py <--- run_experiment.py
                                      |
                                      v
                              OOF预测与训练日志
                                      |
                  +-------------------+-------------------+
                  |                                       |
                  v                                       v
          src/evaluate.py                         benchmark_models.py
                  |                                       |
                  v                                       v
          make_results.py                          resource_usage.csv
                  |
                  v
         指标表、配对差值和图
                  |
                  v
           build_report.py
                  |
                  v
              实验说明DOCX
```

## 顶层Python文件

### `run_experiment.py`

实验训练入口，负责组织全部465次模型训练。

- 从`config.json`读取周数、模型、交叉验证和训练超参数。
- 调用`src.data`读取临床变量、结局标签、交叉验证折和7天聚合PPG序列。
- 对完整模型依次运行前1至前26周实验，共`26 × 5折 × 3种子 = 390`次训练。
- 对MLP、GRU、LSTM、Transformer和普通时序GAT运行完整26周比较，共`5 × 5折 × 3种子 = 75`次训练。
- 第26周完整模型结果直接复用于六模型比较，不重复训练。
- 每完成一个周数或一个比较模型便立即写入结果；重新运行时自动跳过已完成条件，支持中断续跑。

输出为`weekly_oof_predictions.csv`、`model_oof_predictions.csv`、`fold_metrics.csv`和`run_config.json`。

### `make_results.py`

实验统计汇总和作图入口，在训练全部完成后运行。

- 检查逐周预测是否为`26 × 387 = 10,062`行，六模型预测是否为`6 × 387 = 2,322`行。
- 调用`src.evaluate`计算ROC-AUC、AUPRC、敏感度、特异度、F1、准确率和Brier score。
- 使用2,000次分层患者Bootstrap计算95%置信区间。
- 计算第26周相对前1至前25周的配对差值，以及完整模型相对五种比较模型的配对差值。
- 生成逐周性能曲线和六模型比较图。
- 检查预测数量、概率范围、标签一致性和训练记录数。

输出为`weekly_summary_metrics.csv`、`model_summary_metrics.csv`、两个配对差值CSV、`quality_checks.json`和两张PNG图。

### `benchmark_models.py`

六种模型的CPU资源测量脚本，不重新评价预测性能。

- 按`config.json`构建六种完整26周模型。
- 统计可训练参数量。
- 使用单患者、批量大小为1的输入预热20次，再重复推理100次。
- 报告CPU单例推理时间的中位数、四分位距和进程RSS。
- CPU环境不测量GPU显存。

输出为`results/resource_usage.csv`。进程RSS包含Python、PyTorch及已加载依赖，不等同于模型独占内存。

### `build_report.py`

实验说明DOCX生成脚本，不训练模型，也不重新计算统计指标。

- 读取逐周指标、模型指标、配对差值和资源测量结果。
- 插入逐周性能曲线与六模型比较图。
- 生成实验目的、数据与结局、实验设计、训练和统计方法、结果、计算成本、限制及附录。
- 将1至26周完整模型性能整理为附录表。

输出为`report/实验三_逐周减量与模型比较_实验说明.docx`。

## `src`核心模块

### `src/data.py`

负责数据读取、临床变量分组、折内预处理和变长前缀构建。

- `CLINICAL_GROUPS`：将临床变量分为人口学及生活方式、房颤及既往病史、实验室/超声/心电、手术及用药四组。
- `PPG_FEATURES`：定义每个7天节点使用的13个PPG特征。
- `load_common()`：读取患者基线表和Holter结局表，返回基线变量、二分类标签和交叉验证折编号。
- `load_ppg()`：读取PPG表并检查每位患者是否有26个窗口。
- `GroupPreprocessor`：处理一组临床变量，包括数值变量填补和标准化、分类变量填补和独热编码。
- `Data`：统一封装患者ID、临床矩阵、临床组边界、PPG张量、掩码、标签和周数。
- `FoldPreprocessor.fit()`：只在当前训练患者上拟合预处理参数，避免验证集或测试集信息泄漏。
- `FoldPreprocessor.transform()`：按指定周数截取PPG前缀，生成`患者数 × 周数 × 13`的PPG张量及掩码。

Holter负荷数值不会进入临床矩阵或PPG张量，本文件只读取二分类标签和预设折编号。

### `src/models.py`

定义临床编码器、PPG编码器、池化层和六种模型。

- `ClinicalRelationAttention`：完整模型的临床先验编码器。先在四类变量内部计算变量注意力，再计算四类关系表示之间的注意力，输出患者级临床表示。
- `DirectClinical`：比较模型的直接临床投影层。
- `MaskPool`：带掩码的注意力池化，缺失时间节点不参与汇总。
- `TemporalGAT`：加入时间位置编码，仅允许自身和相邻周连接，经过多头注意力和池化得到患者级PPG表示。
- `SequenceEncoder`：统一实现GRU、LSTM和Transformer三种PPG序列编码器。
- `ComparisonModel`：六种模型的统一入口，输出每位患者一个logit。

| `kind` | 临床处理 | PPG处理 | 融合方式 |
|---|---|---|---|
| `mlp` | 预处理临床向量 | 展平PPG及掩码 | 全部拼接后输入MLP |
| `gru` | 直接投影 | GRU | 患者级表示拼接 |
| `lstm` | 直接投影 | LSTM | 患者级表示拼接 |
| `transformer` | 直接投影 | Transformer | 患者级表示拼接 |
| `temporal_gat` | 直接投影 | 相邻周时序GAT | 患者级表示拼接 |
| `full_model` | 临床关系注意力 | 相邻周时序GAT | 患者级表示拼接 |

### `src/train.py`

负责随机性控制、模型拟合、早停、阈值确定、外层交叉验证和折外预测汇总。

- `seed_all()`：同步设置Python、NumPy和PyTorch随机种子。
- `threshold()`：根据内部验证集寻找Youden指数最大的分类阈值。
- `fit()`：构建指定模型，使用加权二元交叉熵和AdamW训练，以验证集ROC-AUC早停，并恢复最佳参数。
- `run_condition()`：完成一个“模型+输入周数”条件的五折交叉验证和三个随机种子训练。

`run_condition()`先划分外层测试折，再从其余患者中划分内部训练集和验证集；预处理器只在内部训练集拟合。每个随机种子独立训练并对测试折预测，最终对同一患者的三个概率和阈值分别取均值，形成OOF记录。

### `src/evaluate.py`

提供性能指标和Bootstrap统计函数。

- `metrics()`：计算ROC-AUC、AUPRC、敏感度、特异度、F1、准确率和Brier score。
- `stratified_bootstrap_indices()`：分别在阳性和阴性患者中有放回抽样，维持每次Bootstrap的类别数量。
- `summarize()`：计算点估计及Bootstrap百分位法95%置信区间。
- `paired_difference()`：使用相同患者Bootstrap索引计算两个模型ROC-AUC和AUPRC的配对差值。

## 测试文件

### `tests/test_data.py`

验证1周、13周和26周前缀截取正确、每位患者均有26个PPG窗口，并检查出院和6个月Holter负荷字段没有进入模型输入。

### `tests/test_models.py`

验证六种模型的输入输出形状、完整模型对1周/13周/26周输入的支持，以及被掩码节点的数值变化不会影响模型输出。

### `tests/test_training.py`

使用小样本和两轮训练执行快速闭环测试，验证数据预处理、模型训练、验证预测和概率输出可以正常运行。

### `tests/test_outputs.py`

验证最终产物：逐周OOF预测10,062行、六模型OOF预测2,322行、训练日志465行、质量检查全部通过，并检查DOCX的必要章节、表格、图片和占位符。

## 配置文件

`config.json`集中控制实验参数。

| 参数 | 当前设置 | 含义 |
|---|---:|---|
| `prefix_weeks` | 1-26 | 完整模型逐周输入长度 |
| `comparison_models` | 6种 | 完整26周比较模型 |
| `outer_folds` | 5 | 外层交叉验证折数 |
| `training_seeds` | 3个 | 每折独立训练种子 |
| `validation_fraction` | 0.2 | 内部验证集比例 |
| `max_epochs` | 120 | 最大训练轮数 |
| `early_stopping_patience` | 15 | 早停耐心值 |
| `hidden_dim` | 32 | 隐藏表示维度 |
| `attention_heads` | 2 | 注意力头数 |
| `dropout` | 0.25 | Dropout比例 |
| `learning_rate` | 0.001 | AdamW学习率 |
| `weight_decay` | 0.0001 | AdamW权重衰减 |
| `batch_size` | 64 | 训练批量大小 |
| `bootstrap_resamples` | 2000 | Bootstrap次数 |

## 输入数据

```text
../outputs/evoaf_synthetic_387/intermediate/
├── patient_baseline.csv
├── holter_outcome.csv
└── ppg_7day.csv
```

当前代码包含387例患者和26个PPG窗口的固定数量检查。如更换数据规模，需要同步修改这些检查，不能只替换CSV。

## 推荐运行顺序

在本目录中执行：

```powershell
# 1. 测试
..\experiment_1_information_source_ablation\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 2. 训练；中断后重复运行可续跑
..\experiment_1_information_source_ablation\.venv\Scripts\python.exe run_experiment.py

# 3. 统计与作图
..\experiment_1_information_source_ablation\.venv\Scripts\python.exe make_results.py

# 4. CPU资源测量
..\experiment_1_information_source_ablation\.venv\Scripts\python.exe benchmark_models.py

# 5. 生成DOCX
..\experiment_1_information_source_ablation\.venv\Scripts\python.exe build_report.py

# 6. 最终产物检查
..\experiment_1_information_source_ablation\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 结果解释边界

- 本实验使用模拟的387例患者数据，主要验证实验流程、模型实现、数据隔离和统计报告。
- 模拟数据性能不能作为模型临床有效性证据。
- 逐周模型均预测同一个6个月结局，并非分别预测各周结局。
- 逐周曲线不用于事后定义“最早稳定周数”。
- 完整模型是否优于比较模型必须由OOF结果和配对置信区间判断，不预设结论。
- `make_results.py`必须在全部训练条件完成后运行，否则数量检查会主动报错。
