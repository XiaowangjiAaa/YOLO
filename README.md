# YOLO11 + SCSegamba Strict-Aligned (for YOLO11n purpose)

你说得对：不应该拍脑袋替换，而要做**放哪、放几个、怎么扫描**的系统消融。

## 我们在 YOLO 里可控替换的模块

可选 stage：`enc1, enc2, enc3, enc4, up1, up2, up3`

当前默认（平衡速度与精度）：
- `enc2, enc3, up3` 使用 `C2fSAVSS`
- 其他 stage 使用 `C2fLite`

可通过训练参数直接控制：
- `--savss-stages "enc2,enc3,up3"`
- `--savss-n 1`（每个被替换 stage 里 SAVSS block 个数）
- `--scan-impl fast|ssm`

## 扫描方式

- `fast`：向量化方向聚合（速度快，显存友好）
- `ssm`：递推状态扫描（更高保真，但慢且占显存）

## 严格对齐训练参数（SCSegamba风格）

- `--dataroot`
- `--batch_size`
- `--epochs`
- `--lr`
- `--BCELoss_ratio`
- `--DiceLoss_ratio`
- `--lr_scheduler`
- `--model_path`

## 推荐先跑的“快版”基线

```bash
python scripts/train_yolo11_savss.py \
  --dataroot /path/to/crack_dataset \
  --batch_size 8 \
  --epochs 200 \
  --lr 0.001 \
  --BCELoss_ratio 0.5 \
  --DiceLoss_ratio 0.5 \
  --lr_scheduler PolyLR \
  --base-ch 16 \
  --savss-stages enc2,enc3,up3 \
  --savss-n 1 \
  --scan-impl fast \
  --no-deep-supervision \
  --amp \
  --model_path runs/yolo11n_savss_fast
```

## 消融实验（重点）

- 提前停止：`--early-stop-patience 50`（若 50 个 epoch 无提升则自动停止）

提供了网格消融脚本和配置：

- 配置：`configs/savss_ablation.json`
- 脚本：`scripts/ablate_savss.py`

先看命令不执行：

```bash
python scripts/ablate_savss.py --config configs/savss_ablation.json --dry-run
```

执行消融：

```bash
python scripts/ablate_savss.py --config configs/savss_ablation.json
```

## 预测

```bash
python scripts/predict_yolo11_savss.py \
  --weights runs/yolo11n_savss_fast/best.pt \
  --source /path/to/images
```

## 对齐检查表

见 `ALIGNMENT.md`。
