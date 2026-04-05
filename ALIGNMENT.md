# SCSegamba Strict Alignment Checklist

## 已对齐

- 训练损失参数命名：`BCELoss_ratio` + `DiceLoss_ratio`
- 学习率调度器命名：`lr_scheduler`，默认 `PolyLR`
- 数据目录风格：`train_img/train_lab`, `val_img/val_lab`
- 训练脚本主入口参数风格：`dataroot`, `epochs`, `batch_size`, `model_path`

## 增强对齐

- 深监督辅助头训练（可开关）
- 训练日志与可视化输出
- AMP 与可选优化器（AdamW/SGD）

## 仍可能存在差异（后续迭代）

- 原版 SAVSS 的底层实现细节（若其外部依赖中有更底层算子）
- 原项目完整数据增强细节（若使用外部 pipeline）
- 原项目可能存在的特定后处理/评估脚本
