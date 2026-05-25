# -*- coding: utf-8 -*-

import os
import json
import torch
import torchvision
from PIL import Image
from colorama import Fore
from src.model import PlainCNN


# =========================
# 1. 基础配置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN + "当前设备：" + str(device))

# 你的测试集目录
TEST_DIR = "../data/test/test"

# 类别映射文件
CLASS_JSON_PATH = "./outputs/class_to_idx.json"

# 你要加载的模型路径
# 比如你训练了 10 轮，最后一轮可能是 model_9.pth
MODEL_PATH = "model_8.pth"


# =========================
# 2. 加载类别映射
# =========================

with open(CLASS_JSON_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

# 反转成：编号 -> 类别名
idx_to_class = {v: k for k, v in class_to_idx.items()}

print(Fore.GREEN + "类别数量：" + str(len(idx_to_class)))


# =========================
# 3. 图片预处理
# =========================

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])


# =========================
# 4. 加载模型
# =========================

# 因为你训练时用的是 torch.save(model, xxx.pth)
# 所以这里直接 torch.load
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

model = model.to(device)
model.eval()

print(Fore.GREEN + "模型加载完成：" + MODEL_PATH)


# =========================
# 5. 单张图片预测函数
# =========================

def predict_one_image(image_path):
    image = Image.open(image_path).convert("RGB")

    image = transform_test(image)

    # 增加 batch 维度
    # 原来是 3 x 256 x 256
    # 变成 1 x 3 x 256 x 256
    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():
        output = model(image)

        # 计算概率
        prob = torch.softmax(output, dim=1)

        # 取最大概率类别
        pred_prob, pred_idx = torch.max(prob, dim=1)

    pred_idx = pred_idx.item()
    pred_prob = pred_prob.item()

    pred_class = idx_to_class[pred_idx]

    return pred_class, pred_prob


# =========================
# 6. 批量预测测试集
# =========================

def predict_test_dir(test_dir):
    results = []

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    for file_name in os.listdir(test_dir):
        file_path = os.path.join(test_dir, file_name)

        if not os.path.isfile(file_path):
            continue

        if not file_name.lower().endswith(tuple(image_extensions)):
            continue

        try:
            pred_class, pred_prob = predict_one_image(file_path)

            results.append({
                "file_name": file_name,
                "pred_class": pred_class,
                "confidence": pred_prob
            })

            print(
                Fore.GREEN +
                f"图片：{file_name} | 预测类别：{pred_class} | 置信度：{pred_prob:.4f}"
            )

        except Exception as e:
            print(Fore.RED + f"图片预测失败：{file_name}，错误：{e}")

    return results


# =========================
# 7. 主函数
# =========================

if __name__ == "__main__":
    results = predict_test_dir(TEST_DIR)

    # 保存预测结果
    save_path = "./outputs/test_predictions.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(Fore.GREEN + "测试集预测完成，结果已保存：" + save_path)