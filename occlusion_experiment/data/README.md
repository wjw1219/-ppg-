# 输入数据格式

将真实数据放在项目根目录的 `data/` 文件夹中。默认需要两个文件：

## `cohort.csv`

每位患者一行，至少包含：

```text
patient_id,label,clinical_feature_1,clinical_feature_2,...
```

`label` 必须是 0/1，表示预先定义的 AF-burden 变化方向。临床变量列可以在 `config.json` 的 `clinical_columns` 中显式列出；留空时，程序会自动选择除 `patient_id` 和 `label` 外的数值列。

## `ppg_weekly.csv`

每位患者每个周次一行，至少包含：

```text
patient_id,week,ppg_feature_1,ppg_feature_2,...
```

`week` 取 1–26。PPG 特征列可以在 `config.json` 的 `ppg_columns` 中显式列出；留空时，程序会自动选择除 `patient_id` 和 `week` 外的数值列。

缺失周次允许不提供该行，程序会将该节点标记为无效并在训练和推理时使用 attention mask。重复的 `patient_id-week` 组合会直接报错，避免无意聚合导致数据泄漏。

CSV 也可以替换为同名 `.parquet` 或 `.xlsx` 文件；文件扩展名需要同步修改 `config.json`。
