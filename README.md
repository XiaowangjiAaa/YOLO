# YOLO11 + SCSegamba Core Module Integration (Prototype)

你提到“核心其实是 SAVSS”，本次已把实现重点从单独 GBC 升级为 **SAVSS 风格块**，并补齐了训练/预测/数据集准备脚本（参考 SCSegamba 的 `main.py + test.py + datasets` 使用方式）。

## 当前实现的核心

1. **GBC (Gated Bottleneck Convolution)**：做局部形态/纹理增强。  
2. **SASS2D (Structure-Aware Scanning Strategy)**：通过四方向扫描（左右/上下）聚合长程连续结构信息。  
3. **SAVSSBlock**：将 `GBC + SASS2D` 串联并残差连接，作为可重复堆叠的核心单元。  
4. **C2fSAVSS**：将 YOLO C2f 中的内部块替换为 `SAVSSBlock`，直接可用于 backbone/neck。  

## 与 SCSegamba 的对齐点

- SCSegamba 在 README 中给出：训练入口 `main.py`、推理入口 `test.py`、并采用统一参数风格。  
- 其数据组织示例是 `train_img/train_lab`、`val_img/val_lab` 等成对目录。  
- 本仓库据此提供：
  - `scripts/prepare_crack_dataset.py`：把上述 mask 数据转换为 YOLO 分割标签。  
  - `scripts/train_yolo11_savss.py`：一体化“先准备数据再训练”。  
  - `scripts/predict_yolo11_savss.py`：加载权重进行预测。  

## 数据集目录（输入）

```text
<dataset_root>/
  train_img/
  train_lab/
  val_img/
  val_lab/
  test_img/   # 可选
  test_lab/   # 可选
```

`*_lab` 是二值 mask（>0 视为裂缝前景）。

## 一键准备 YOLO 数据集

```bash
python scripts/prepare_crack_dataset.py \
  --src /path/to/crack_dataset \
  --dst datasets/crack_yolo
```

转换后会生成：

- `datasets/crack_yolo/images/{train,val,test}`
- `datasets/crack_yolo/labels/{train,val,test}`
- `datasets/crack_yolo/data.yaml`

## 训练

```bash
python scripts/train_yolo11_savss.py \
  --dataset-root /path/to/crack_dataset \
  --prepared-dataset datasets/crack_yolo \
  --model-cfg ultralytics/cfg/models/11/yolo11-scgb.yaml \
  --epochs 200 --imgsz 640 --batch 16 --device 0
```

只做数据准备，不训练：

```bash
python scripts/train_yolo11_savss.py \
  --dataset-root /path/to/crack_dataset \
  --prepare-only
```

## 预测

```bash
python scripts/predict_yolo11_savss.py \
  --weights runs/train/yolo11_savss/weights/best.pt \
  --source /path/to/images_or_video \
  --imgsz 640 --conf 0.25 --device 0
```

## 文件说明

- `ultralytics/nn/modules/scsegamba.py`：`BottConv/GBC/SASS2D/SAVSSBlock/C2fSAVSS`。
- `ultralytics/cfg/models/11/yolo11-scgb.yaml`：示例模型（多处使用 `C2fSAVSS`）。
- `scripts/prepare_crack_dataset.py`：SCSegamba 风格数据读取与 YOLO 标签转换。
- `scripts/train_yolo11_savss.py`：训练入口脚本。
- `scripts/predict_yolo11_savss.py`：预测入口脚本。
- `tests/test_scsegamba_module.py`：核心模块基础 shape 测试。
