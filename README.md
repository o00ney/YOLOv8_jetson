# YOLOv8 Jetson

项目方案见 [PLAN.md](PLAN.md)。官方 Ultralytics 源码通过 Git 子模块固定版本。

[在 Colab 中打开训练 Notebook](https://colab.research.google.com/github/o00ney/YOLOv8_jetson/blob/main/train_colab.ipynb)

先选择 GPU，然后依次运行 Notebook 单元格。Drive 挂载需要交互授权。

- 模型：YOLOv8s，自动下载 COCO 预训练权重 yolov8s.pt。
- 数据：Construction-PPE，11 类；训练 1132 / 验证 143 / 测试 141。图片与压缩包保留在本地，GitHub 不重复上传，Colab 从官方地址下载。
- 初始训练：100 epochs，imgsz=640，batch=16，workers=2；显存不足时减小 batch。
- 结果：Google Drive 的 MyDrive/YOLOv8_jetson/yolov8s_ppe_<UTC时间>/，包含 weights/best.pt、weights/last.pt、results.csv、args.yaml 和图表。
- 训练成功并检查结果后，flush_and_unmount 完成同步，再调用 runtime.unassign() 释放运行时。失败时不会主动断开；Colab 平台自身的资源限制仍可能中断会话。
- Notebook 在本地完成静态检查，尚未在真实 Colab GPU 上训练验证。

本地拉取：

```bash
git clone --recurse-submodules https://github.com/o00ney/YOLOv8_jetson.git
```

Colab 命令（每个代码块为一个单元格）：

```python
%cd /content
!git clone --recurse-submodules https://github.com/o00ney/YOLOv8_jetson.git
%cd /content/YOLOv8_jetson
%pip install -e ./ultralytics
```

```python
from google.colab import drive
drive.mount("/content/drive")
```

```python
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import yaml

dataset_dir = Path("/content/datasets/construction-ppe")
dataset_dir.mkdir(parents=True, exist_ok=True)
archive_path = Path("/content/construction-ppe.zip")
urlretrieve(
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
    archive_path,
)
with ZipFile(archive_path) as archive:
    archive.extractall(dataset_dir)

config = yaml.safe_load(
    Path("ultralytics/ultralytics/cfg/datasets/construction-ppe.yaml").read_text()
)
config["path"] = str(dataset_dir)
config.pop("download", None)
data_yaml = dataset_dir / "data.yaml"
data_yaml.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

for split, expected in (("train", 1132), ("val", 143), ("test", 141)):
    images = list((dataset_dir / "images" / split).glob("*"))
    assert len(images) == expected, (split, len(images))
    assert all(
        (dataset_dir / "labels" / split / (image.stem + ".txt")).is_file()
        for image in images
    )
    print(f"{split}: {len(images)} images, labels verified")
```

```python
import json
import subprocess
from datetime import datetime, timezone
import torch
import ultralytics
from ultralytics import YOLO, settings
from google.colab import drive, runtime

assert torch.cuda.is_available(), "请在 Colab 运行时设置中选择 GPU"
assert Path("/content/drive/MyDrive").is_dir(), "请先挂载 Google Drive"
settings.update({"wandb": False, "comet": False, "mlflow": False})
run_name = "yolov8s_ppe_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
run_dir = Path("/content/drive/MyDrive/YOLOv8_jetson") / run_name
run_dir.mkdir(parents=True, exist_ok=False)

metadata = {
    "ultralytics": ultralytics.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "source_commit": subprocess.check_output(
        ["git", "-C", "ultralytics", "rev-parse", "HEAD"], text=True
    ).strip(),
}
(run_dir / "environment.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print(f"训练结果保存至: {run_dir}")

model = YOLO("yolov8s.pt")  # 自动下载 COCO 预训练权重
model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=2,
    project=str(run_dir.parent),
    name=run_dir.name,
    exist_ok=True,
    save=True,
    plots=True,
)
for filename in ("weights/best.pt", "weights/last.pt", "results.csv", "args.yaml"):
    artifact = run_dir / filename
    assert artifact.is_file() and artifact.stat().st_size > 0, artifact

print(f"训练完成，结果已写入: {run_dir}")
# 仅在训练及结果检查成功后执行；同步失败会抛出异常并保留运行时。
drive.flush_and_unmount(timeout_ms=60000)
runtime.unassign()
```

官方参考：[训练参数](https://docs.ultralytics.com/modes/train/)、[Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/)、[Colab runtime](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py)。
