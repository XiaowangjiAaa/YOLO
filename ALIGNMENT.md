# SCSegamba Strict Alignment Checklist

## 已对齐

- 训练损失参数命名：`BCELoss_ratio` + `DiceLoss_ratio`
- 学习率调度器命名：`lr_scheduler`，默认 `PolyLR`
- 数据目录风格：`train_img/train_lab`, `val_img/val_lab`
- 训练脚本主入口参数风格：`dataroot`, `epochs`, `batch_size`, `model_path`

## 消融能力（本次新增）

- 可控替换位置：`savss_stages`（enc/up 全部可选）
- 可控替换深度：`savss_n`
- 可控扫描实现：`scan_impl = fast | ssm`
- 提供网格消融脚本：`scripts/ablate_savss.py`
- 提供消融配置样例：`configs/savss_ablation.json`

## 为 YOLO11n 目标做的工程化调整

- 选择性替换 SAVSS（不是全网络替换）
- 默认 `scan_impl=fast`
- 默认 `base_ch=16`，默认关闭 deep supervision
- 支持 AMP 与模型参数量打印

## 仍可能存在差异（后续迭代）

- 原版 SAVSS 的底层算子细节（若其外部依赖中有更底层实现）
- 原项目完整数据增强细节
- 原项目可能存在的后处理/评估脚本
