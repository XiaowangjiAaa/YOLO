# YOLO11 + SCSegamba Core Module Integration (Prototype)

你指出的关键问题是对的：如果直接 `pip install ultralytics` 来训练，官方包并不知道我们本地新增的 `SAVSS` 模块。  
所以这版改成 **本地纯 PyTorch 训练/预测流程**，不依赖外部 ultralytics 安装包来解析模型。

## 核心模块（已本地落地）

- `GBC`：门控瓶颈卷积。  
- `SASS2D`：四方向结构扫描。  
- `SAVSSBlock`：`GBC + SASS2D + residual`。  
- `C2fSAVSS`：YOLO 风格容器。  

代码位于 `ultralytics/nn/modules/scsegamba.py`。

## 新增：本地训练/预测栈（不依赖 ultralytics 包）

- `yolo11_savss/model.py`：定义 `YOLO11SAVSSSeg`（encoder-decoder，内部使用 `C2fSAVSS`）。
- `yolo11_savss/data.py`：按 SCSegamba 风格读取 `train_img/train_lab` 等目录。
- `scripts/train_yolo11_savss.py`：本地训练入口（BCE loss，输出 `best.pt/last.pt`）。
- `scripts/predict_yolo11_savss.py`：本地预测入口（输出二值 mask 图）。

## 数据目录（与 SCSegamba 风格一致）

```text
<dataset_root>/
  train_img/
  train_lab/
  val_img/
  val_lab/
  test_img/   # 可选
  test_lab/   # 可选
```

`*_lab` 为二值 mask（>0 视为前景）。

## 训练

```bash
python scripts/train_yolo11_savss.py \
  --dataset-root /path/to/crack_dataset \
  --epochs 100 \
  --batch-size 8 \
  --image-size 512 \
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

## 可选：转换为 YOLO 分割标签

如果你后续仍想接 YOLO 格式数据，也可以用：

```bash
python scripts/prepare_crack_dataset.py \
  --src /path/to/crack_dataset \
  --dst datasets/crack_yolo
```

> 该脚本依赖 `opencv-python`（`cv2`）。
