# YOLO11 + SCSegamba High-Fidelity Replica (Local PyTorch)

这版是“高保真复刻”方向：不再只做轻量近似，而是把核心模块与训练策略向 SCSegamba 靠拢。

## 核心模块

- `GBC`：门控瓶颈卷积。
- `SAVSS2D`：**可学习状态空间方向扫描**（四方向，带 learnable `a/b/c/d` 递推参数），不是简单 `cumsum`。
- `SAVSSBlock`：`GBC + SAVSS2D + FFN`。
- `C2fSAVSS`：YOLO 风格封装。

模块实现：`ultralytics/nn/modules/scsegamba.py`。

## 网络结构

- `yolo11_savss/model.py` 中 `YOLO11SAVSSSeg` 采用 encoder-decoder + 多尺度融合。
- 支持 `deep_supervision`，输出主头 + 辅助头用于训练监督。

## 训练策略（对齐 SCSegamba 思路）

- 混合损失：`BCELoss_ratio * BCE + DiceLoss_ratio * DiceLoss`
- 学习率：`PolyLR`
- 深监督：主输出 + 辅助输出加权（`--aux-weight`）
- 训练进度可视化：
  - tqdm/step 打印
  - `train_log.csv`
  - `val_vis/`（`RGB | GT | Pred`）

## 数据目录（SCSegamba 风格）

```text
<dataset_root>/
  train_img/
  train_lab/
  val_img/
  val_lab/
  test_img/   # 可选
  test_lab/   # 可选
```

## 训练

```bash
python scripts/train_yolo11_savss.py \
  --dataset-root /path/to/crack_dataset \
  --epochs 200 \
  --batch-size 8 \
  --image-size 512 \
  --BCELoss-ratio 0.5 \
  --DiceLoss-ratio 0.5 \
  --poly-power 0.9 \
  --aux-weight 0.4 \
  --device cuda
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

## 可选：转换为 YOLO 标签

```bash
python scripts/prepare_crack_dataset.py \
  --src /path/to/crack_dataset \
  --dst datasets/crack_yolo
```

> 该转换脚本依赖 `opencv-python`（cv2）。
