# YOLO11 + SCSegamba Strict-Aligned Replica (Local PyTorch)

这版是“严格对齐版”：优先对齐 SCSegamba 的训练参数命名、损失组合、调度策略和数据组织方式。

## 核心模块

- `GBC`：门控瓶颈卷积。
- `SAVSS2D`：可学习状态空间方向扫描（四方向，learnable `a/b/c/d`）。
- `SAVSSBlock`：`GBC + SAVSS2D + FFN`。
- `C2fSAVSS`：YOLO 风格封装。

实现文件：`ultralytics/nn/modules/scsegamba.py`。

## 模型结构

`yolo11_savss/model.py` 中 `YOLO11SAVSSSeg` 使用 encoder-decoder + 多尺度融合，并支持深监督输出。

## 严格对齐训练脚本

`python scripts/train_yolo11_savss.py`

支持 SCSegamba 风格参数：

- `--dataroot`
- `--batch_size`
- `--epochs`
- `--lr`
- `--BCELoss_ratio`
- `--DiceLoss_ratio`
- `--lr_scheduler` (`PolyLR`/`Cosine`)
- `--model_path`

并保留本地别名参数（兼容旧调用）：`--dataset-root`、`--batch-size`、`--BCELoss-ratio` 等。

## 训练策略

- `HybridLoss = BCELoss_ratio * BCE + DiceLoss_ratio * DiceLoss`
- 默认 `PolyLR`
- 深监督（可开关）：`--deep-supervision / --no-deep-supervision`
- 辅助头损失权重：`--aux-weight`

## 训练进度可视化

- 控制台：tqdm/step 打印
- 文件：`train_log.csv`
- 可视化：`val_vis/`（`RGB | GT | Pred`）

## 数据目录

```text
<dataset_root>/
  train_img/
  train_lab/
  val_img/
  val_lab/
  test_img/   # 可选
  test_lab/   # 可选
```

## 示例训练命令（严格对齐风格）

```bash
python scripts/train_yolo11_savss.py \
  --dataroot /path/to/crack_dataset \
  --epochs 200 \
  --batch_size 8 \
  --lr 0.001 \
  --BCELoss_ratio 0.5 \
  --DiceLoss_ratio 0.5 \
  --lr_scheduler PolyLR \
  --model_path runs/local_yolo11_savss
```

## 预测

```bash
python scripts/predict_yolo11_savss.py \
  --weights runs/local_yolo11_savss/best.pt \
  --source /path/to/images_or_dir \
  --image-size 512 \
  --threshold 0.5 \
  --device cuda
```

## 对齐检查表

见 `ALIGNMENT.md`。

## 可选：转换为 YOLO 标签

```bash
python scripts/prepare_crack_dataset.py \
  --src /path/to/crack_dataset \
  --dst datasets/crack_yolo
```

> 该转换脚本依赖 `opencv-python`（cv2）。
