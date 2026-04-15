# YOLO11 + SCSegamba Strict-Aligned (for YOLO11n purpose)

你提到的关键升级点是：把 SSM 从 **static 参数** 升级为 **dynamic/selective 参数**（更接近 Mamba 思想）。

## Dynamic SAVSS（Mamba-like）

在 `scan_impl=ssm` 下新增：

- `--ssm-mode static`：固定参数 `a,b,c,d`
- `--ssm-mode dynamic`：输入相关 `a(x),b(x),c(x),d(x)`（通过 1x1 Conv1d 产生）

也就是从：
- Static SSM → Dynamic Selective SSM

## 可控替换模块（用于消融）

可选 stage：`enc1, enc2, enc3, enc4, up1, up2, up3`

关键参数：
- `--savss-stages`
- `--savss-n`
- `--scan-impl fast|ssm`
- `--ssm-mode static|dynamic`（仅 ssm 下生效）

## 推荐主线（edge）

- 主线仍建议先用 `fast`
- 仅在小规模对比里打开 `ssm static/dynamic` 检查 dynamic 是否有增益

## 消融实验配置

`configs/savss_ablation.json` 当前包含：

1. 你要求的 fast 6+1 组合
2. 额外的 dynamic-vs-static 对比：
   - `ssm_static_n1_enc3_up3`
   - `ssm_dynamic_n1_enc3_up3`
   - `ssm_static_n2_enc3_up3`
   - `ssm_dynamic_n2_enc3_up3`

运行：

```bash
python scripts/ablate_savss.py --config configs/savss_ablation.json --dry-run
python scripts/ablate_savss.py --config configs/savss_ablation.json
```

## 统计输出

- 每个 run：`run_summary.csv`（`best_miou / params / est_flops`）
- 消融总表：`runs/ablation_savss/ablation_summary.csv`

## 训练示例

```bash
python scripts/train_yolo11_savss.py \
  --dataroot /path/to/crack_dataset \
  --savss-stages enc3,up3 \
  --savss-n 1 \
  --scan-impl fast \
  --ssm-mode dynamic \
  --early-stop-patience 50 \
  --amp
```
