# 程序说明

## 文件

- script/HMSA-ViM-YOLO.py：模型、数据读取、训练、验证、测试、鲁棒性评估与结果可视化完整程序。

## 数据目录

数据根目录需按如下结构组织（以 /data/processed_dataset_HMSA-ViM-YOLO 为例）：

```bash
processed_dataset_HMSA-ViM-YOLO/
  pdt_yolo/                       # PDT 数据集（YOLO 格式）
    images/
      train/
      val/
      test/
    labels/
      train/
      val/
      test/
  banana_fusarium/                # 香蕉枯萎病数据集
    banana_manifest.csv           # 划分清单
    rgb_yolo/                     # RGB 图像
    ms_npz/                       # 9 通道多光谱张量
    labels/                       # YOLO 标签
```

图像和掩膜需同名，例如 `001.png` 对应 `001.png`。掩膜像素值应为类别索引：`0,1,2,3`。若做二分类，可设置 `--num-classes 2`，程序会把 `0/255` 掩膜映射为 `0/1`。

香蕉数据通过 banana_manifest.csv 指定：

```bash
split,image,npz,label
train,rgb_yolo/a.png,ms_npz/a.npz,labels/a.txt
val,rgb_yolo/b.png,ms_npz/b.npz,labels/b.txt
test,rgb_yolo/c.png,ms_npz/c.npz,labels/c.txt
```

## 安装

```bash
conda create -n HMSA-ViM-YOLO python=3.10 -y
conda activate HMSA-ViM-YOLO

pip install torch torchvision numpy pillow
pip install thop
```

`mamba-ssm` 和 `causal-conv1d` 用于大尺寸 SS2D 加速。

## 数据集组合
--dataset 可选择一个或多个，名称固定为 pdt、banana_rgb、banana_ms：

| 组合 | 输入通道 | 前景类别 |
| --- | ---: | --- |
| `--dataset pdt` | 3 | `pdt_unhealthy` |
| `--dataset banana_rgb` | 3 | `banana_healthy_plant`, `banana_fusarium_wilt_plant`（可切换 `disease_only`） |
| `--dataset banana_ms` | 9 | 同上 |
| `--dataset pdt banana_ms` | 9 | 上述类别合并 |

香蕉标签策略由 --banana-label-policy {two_class, disease_only} 控制：
two_class：健康株和枯萎株都作为前景类；
disease_only：仅枯萎株作为前景类，健康株作为背景负样本。

## 训练

单数据集训练（PDT）：：
```bash
python scripts/HMSA-ViM-YOLO.py train \
  --dataset pdt \
  --data-root /data/processed_dataset_HMSA-ViM-YOLO \
  --output-dir runs_HMSA-ViM-YOLO \
  --epochs 80 \
  --lr 1e-4 \
  --batch-size 4 \
  --width 48 \
  --depth 2 \
  --scheduler onecycle \
  --optimizer adamw \
  --train-aug \
  --amp
```

多数据集联合训练（PDT + 香蕉多光谱，数据源均衡采样）：
```bash
python scripts/HMSA-ViM-YOLO.py train \
  --dataset pdt banana_ms \
  --data-root /data/processed_dataset_HMSA-ViM-YOLO \
  --output-dir runs_HMSA-ViM-YOLO \
  --epochs 80 \
  --lr 1e-4 \
  --batch-size 4 \
  --width 48 \
  --depth 2 \
  --source-balanced \
  --source-sampling pdt:0.7,banana_ms:0.3 \
  --train-aug \
  --amp
```

输出目录：
```bash
runs_vmambapet/<dataset_key>/<timestamp>/
  best.pt
  latest.pt
  run_config.json
  train_log.jsonl
  <timestamp>_epochXXX_map50_XXXX_val.pt
```
若指定 --ablation 且不是 full，目录改为 runs_HMSA-ViM-YOLO/<dataset_key>/<ablation>/<timestamp>/。

## 测试

```bash
python scripts/HMSA-ViM-YOLO.py test \
  --dataset pdt \
  --data-root /data/processed_dataset_HMSA-ViM-YOLO \
  --weights runs_HMSA-ViM-YOLO/pdt/20260629-172032/best.pt \
  --ablation full \
  --split test \
  --save-result-images
```

## 鲁棒性测试

单图：

```bash
python scripts/HMSA-ViM-YOLO_experiment.py robust \
  --dataset pdt \
  --data-root /data/processed_dataset_HMSA-ViM-YOLO \
  --weights runs_HMSA-ViM-YOLO/pdt/full/20260629-172032/best.pt \
  --ablation full \
  --split test
```

## 消融实验
可用消融方案：
| 名称 | 说明 |
| --- | --- |
| `full` | 完整模型（stress prior + directional scan + P2） |
| `baseline` | 去掉 stress、directional 与 P2 |
| `no_stress` | 去掉光谱先验 |
| `no_directional` | 去掉多方向扫描 |
| `no_p2` | 去掉 P2 特征层 |
| `no_aug` | 保留模块但关闭训练增强 |
| `stress_only` | 只保留光谱先验 |
| `directional_only` | 只保留多方向扫描 |
替换"--ablation full"以使用

## 参考资料

- torchvision Faster R-CNN：https://pytorch.org/vision/stable/models/faster_rcnn.html
- Mamba 官方仓库：https://github.com/state-spaces/mamba
- thop（FLOPs 统计）：https://github.com/Lyken17/pytorch-OpCounter
- PDT数据集：https://github.com/RuiXing123/PDT_CWC_YOLO-DP
