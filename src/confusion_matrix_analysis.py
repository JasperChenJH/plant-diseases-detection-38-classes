# -*- coding: utf-8 -*-

import os
import json
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from colorama import Fore
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report

# 必须导入你的模型类
from src.model import PlainCNN


# =========================
# 1. 基础配置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN + "当前设备：" + str(device))

ROOT_DIR = "../data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
VALID_DIR = os.path.join(ROOT_DIR, "valid")

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 改成你准确率最高的模型
MODEL_PATH = "model_8.pth"

CLASS_JSON_PATH = "./outputs/class_to_idx.json"

BATCH_SIZE = 16


# =========================
# 2. 加载类别映射
# =========================

with open(CLASS_JSON_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

print(Fore.GREEN + f"类别数量：{len(class_names)}")


# =========================
# 3. 验证集预处理
# =========================

transform_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])

valid_dataset = ImageFolder(VALID_DIR, transform=transform_valid)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(Fore.GREEN + f"验证集样本数：{len(valid_dataset)}")


# =========================
# 4. 加载模型
# =========================

# 因为你训练时 torch.save(model, "model_x.pth") 保存的是整个模型
# PyTorch 2.6 后需要 weights_only=False
model = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = model.to(device)
model.eval()

print(Fore.GREEN + "模型加载完成：" + MODEL_PATH)


# =========================
# 5. 在验证集上预测
# =========================

all_labels = []
all_preds = []

correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(valid_loader, desc="验证集中预测"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = correct / total

print(Fore.GREEN + f"验证集准确率：{acc:.4f}")
print(Fore.GREEN + f"验证集准确率百分比：{acc * 100:.2f}%")


# =========================
# 6. 计算混淆矩阵
# =========================

cm = confusion_matrix(all_labels, all_preds)

# 保存原始混淆矩阵
cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_csv_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path, encoding="utf-8-sig")

print(Fore.GREEN + "混淆矩阵 CSV 已保存：" + cm_csv_path)


# =========================
# 7. 画混淆矩阵图片
# =========================

plt.figure(figsize=(18, 16))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(class_names))

# 类别名太长，图上放数字编号更清楚
plt.xticks(tick_marks, tick_marks, rotation=90)
plt.yticks(tick_marks, tick_marks)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()

cm_img_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(cm_img_path, dpi=300)
plt.close()

print(Fore.GREEN + "混淆矩阵图片已保存：" + cm_img_path)


# =========================
# 8. 输出编号和类别对应关系
# =========================

label_map_path = os.path.join(OUTPUT_DIR, "label_index_mapping.txt")

with open(label_map_path, "w", encoding="utf-8") as f:
    for i, name in enumerate(class_names):
        f.write(f"{i}: {name}\n")

print(Fore.GREEN + "类别编号映射已保存：" + label_map_path)


# =========================
# 9. 找出最容易混淆的类别
# =========================

confusions = []

for true_idx in range(len(class_names)):
    for pred_idx in range(len(class_names)):
        if true_idx == pred_idx:
            continue

        count = cm[true_idx, pred_idx]

        if count > 0:
            true_class = class_names[true_idx]
            pred_class = class_names[pred_idx]

            confusions.append({
                "true_idx": true_idx,
                "pred_idx": pred_idx,
                "true_class": true_class,
                "pred_class": pred_class,
                "count": int(count)
            })

# 按误判次数从大到小排序
confusions = sorted(confusions, key=lambda x: x["count"], reverse=True)

top_k = 20
top_confusions = confusions[:top_k]

print("\n" + Fore.YELLOW + f"最容易混淆的前 {top_k} 组类别：")

for item in top_confusions:
    print(
        Fore.YELLOW +
        f"真实类别 [{item['true_idx']}] {item['true_class']} "
        f"被误判为 [{item['pred_idx']}] {item['pred_class']} "
        f"次数：{item['count']}"
    )

# 保存容易混淆类别
confusion_pairs_path = os.path.join(OUTPUT_DIR, "top_confusions.csv")

pd.DataFrame(top_confusions).to_csv(
    confusion_pairs_path,
    index=False,
    encoding="utf-8-sig"
)

print(Fore.GREEN + "易混淆类别 CSV 已保存：" + confusion_pairs_path)


# =========================
# 10. 保存分类报告
# =========================

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(Fore.GREEN + "分类报告已保存：" + report_path)

print("\n" + report)