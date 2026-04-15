# SCSegamba Strict Alignment Checklist

## 已对齐

- 训练损失参数命名：`BCELoss_ratio` + `DiceLoss_ratio`
- 学习率调度器命名：`lr_scheduler`，默认 `PolyLR`
- 数据目录风格：`train_img/train_lab`, `val_img/val_lab`
- 训练脚本主入口参数风格：`dataroot`, `epochs`, `batch_size`, `model_path`

## SSM 设计升级（本次）

- `scan_impl=ssm` 支持两种模式：
  - `ssm_mode=static`（固定参数）
  - `ssm_mode=dynamic`（输入相关 selective 参数）
- 提供 dynamic vs static 对比消融配置（见 `configs/savss_ablation.json`）

## 消融能力

- 可控替换位置：`savss_stages`
- 可控替换深度：`savss_n`
- 可控扫描实现：`scan_impl`
- 可控 SSM 模式：`ssm_mode`
- 自动汇总：`ablation_summary.csv`（含 mIoU/params/FLOPs）

## 为 YOLO11n 目标做的工程化调整

- 默认主线为 `fast`
- 默认 `base_ch=16`，默认关闭 deep supervision
- 支持 AMP 与模型参数量打印
- 支持 early-stop（默认 patience=50）
