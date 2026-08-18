# 逐周 PPG 单窗口遮蔽实验

该目录包含真实数据到位后运行单周遮蔽实验的完整代码。实验流程为：

1. 以患者为单位进行五折外层交叉验证；
2. 在每个外层训练折内部再划分验证集，用于 early stopping 和 Youden 阈值确定；
3. 仅使用训练子集拟合临床变量和 PPG 特征的均值、标准差及缺失填补参数；
4. 训练完整 26 周模型，并在外层测试患者上冻结参数；
5. 依次遮蔽第 1–26 周的单个 PPG 节点，保持节点位置编码和周次索引不变，不重新编号；
6. 由每个患者的折外连续预测概率计算 ROC-AUC、AUPRC、Brier score，以及由同一阈值产生的 TP、FN、TN、FP、Sensitivity、Specificity 和 F1；
7. 输出患者级折外预测、逐周指标表、Bootstrap 区间、训练日志及 PDF/SVG 图形。

主要输出文件包括：

- `metrics_by_week.csv`：完整26周参考及第1–26周单周遮蔽的汇总指标；
- `metrics_by_fold.csv`：每个外层折的患者级指标；
- `patient_oof_predictions_long.csv`：每位患者、每个遮蔽周次的折外概率和分类结果；
- `training_log.csv`：每折、每个随机种子的训练轮数、阈值和数据规模；
- `figures/single_week_occlusion_performance.(pdf|svg)` 和 `single_week_occlusion_degradation.(pdf|svg)`：论文图形。

结果表中 `masked_week=0` 表示完整26周输入参考模型；`masked_week=1` 至 `26` 分别表示只遮蔽对应周次。

## 运行方式

在项目根目录执行：

```powershell
python occlusion_experiment/run_single_week_occlusion.py --config occlusion_experiment/config.json
python occlusion_experiment/plot_results.py --results outputs/single_week_occlusion/metrics_by_week.csv --out-dir outputs/single_week_occlusion/figures
```

实际运行前请安装 `requirements.txt` 中的依赖，并把真实数据放入 `data/`。若字段名与默认约定不同，先修改 `config.json`。

代码内置了一个可复现的临床先验 + 因果时序注意力参考实现。如果论文主分析已经有经过冻结的正式模型代码，应将 `run_single_week_occlusion.py` 中的 `ClinicalPriorTemporalGraph` 替换为该正式模型；数据划分、预处理、单周遮蔽和指标计算部分不需要改变。

## 重要分析约束

- 逐周遮蔽是冻结模型推理实验，不针对每个遮蔽周次重新训练模型。
- 所有划分、标准化、缺失填补和阈值选择均在患者级进行。
- ROC-AUC、AUPRC 和 Brier score 使用连续预测概率；Sensitivity、Specificity 和 F1 严格由相同患者集合上的 TP/FN/TN/FP 计算。
- Bootstrap 是患者级折外预测 bootstrap，反映 OOF 预测的不确定性；它不等价于每次重采样都重新训练和重新选择模型。若需要完整重训练 bootstrap，应另行启用计算量更高的重采样方案。
- 程序不会生成模拟结果；数据缺失、字段不匹配或重复 patient-week 会直接报错。
