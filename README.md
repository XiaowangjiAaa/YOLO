# YOLO11 + SCSegamba Strict-Aligned (for YOLO11n purpose)

你的目标是：**精度更高、速度更快、参数更小**。这版按这个目标做了结构调整。

## 我们在 YOLO 中替换了哪些模块

在本地 `YOLO11SAVSSSeg` 里并不是“全替换”，而是**选择性替换**：

- `enc2`、`enc3`、`up3` 使用 `C2fSAVSS`（引入 SCSegamba 思想）
- `enc1`、`enc4`、`up2`、`up1` 使用 `C2fLite`（保持速度和小参数）

这样是为了平衡精度与速度，不让全网络都跑重型扫描。

## 速度/显存优化点

1. `SAVSS2D` 增加 `scan_impl`：
   - `fast`：向量化方向聚合（默认，快）
   - `ssm`：递推状态扫描（更接近高保真，但慢）
2. 默认 `base_ch=16`（更接近 yolo11n 轻量预算）
3. 默认关闭 deep supervision（减少显存和训练耗时）
4. 支持 AMP：`--amp`

## 严格对齐训练参数（SCSegamba风格）

- `--dataroot`
- `--batch_size`
- `--epochs`
- `--lr`
- `--BCELoss_ratio`
- `--DiceLoss_ratio`
- `--lr_scheduler`
- `--model_path`

## 训练示例（推荐先跑快版）

```bash
python scripts/train_yolo11_savss.py \
  --dataroot /path/to/crack_dataset \
  --batch_size 8 \
  --epochs 200 \
  --lr 0.001 \
  --BCELoss_ratio 0.5 \
  --DiceLoss_ratio 0.5 \
  --lr_scheduler PolyLR \
  --scan-impl fast \
  --model_path runs/yolo11n_savss_fast \
  --amp
```

## 预测

```bash
python scripts/predict_yolo11_savss.py \
  --weights runs/yolo11n_savss_fast/best.pt \
  --source /path/to/images
```

## 对齐检查表

见 `ALIGNMENT.md`。
