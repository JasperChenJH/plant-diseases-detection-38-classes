# -*- coding: utf-8 -*-

import os
import json
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from colorama import Fore

from src.model import PlainCNN


# =========================
# 1. 基础配置
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN + "当前设备：" + str(device))

# 你的模型路径
MODEL_PATH = "model_8.pth"

# 类别映射
CLASS_JSON_PATH = "./outputs/class_to_idx.json"

# 需要解释的图片路径
IMAGE_PATH = "../data/test/test/PotatoEarlyBlight1.JPG"

# Grad-CAM 输出路径
SAVE_PATH = "./outputs/grad_cam_result.jpg"

os.makedirs("./outputs", exist_ok=True)


# =========================
# 2. 读取类别映射
# =========================

with open(CLASS_JSON_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}

print(Fore.GREEN + "类别数量：" + str(len(idx_to_class)))


# =========================
# 3. 图片预处理
# =========================

# 注意：这里要和你训练时保持一致
# 你训练时是 Resize((256, 256)) + ToTensor()
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor()
])


def load_image(image_path):
    """
    读取图片，并返回：
    1. 原始 PIL 图片
    2. 模型输入 tensor
    """
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image)

    # 增加 batch 维度
    # [3, 256, 256] -> [1, 3, 256, 256]
    input_tensor = input_tensor.unsqueeze(0)

    return image, input_tensor


# =========================
# 4. 加载模型
# =========================

model = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model = model.to(device)
model.eval()

print(Fore.GREEN + "模型加载完成：" + MODEL_PATH)


# =========================
# 5. 选择 Grad-CAM 的目标层
# =========================

"""
你的 PlainCNN 是 nn.Sequential，结构大概是：

model.module[0]  Conv2d 3 -> 32
...
model.module[27] Conv2d 256 -> 512
...
model.module[30] Conv2d 512 -> 512

Grad-CAM 通常选择最后一个卷积层。
你的最后一个卷积层是 model.module[30]。
"""

target_layer = model.module[30]


# =========================
# 6. Grad-CAM 核心类
# =========================

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # 注册前向 hook，保存特征图
        self.forward_hook = self.target_layer.register_forward_hook(
            self.save_activation
        )

        # 注册反向 hook，保存梯度
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, input, output):
        """
        保存目标卷积层的输出特征图。
        """
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """
        保存目标类别对目标卷积层输出的梯度。
        """
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        生成 Grad-CAM 热力图。

        input_tensor: [1, 3, 256, 256]
        class_idx: 指定解释哪个类别；如果为 None，就解释模型预测类别
        """
        input_tensor = input_tensor.to(device)

        # 前向传播
        output = self.model(input_tensor)

        # 预测概率
        probabilities = torch.softmax(output, dim=1)

        # 如果没有指定类别，就取预测概率最大的类别
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probabilities[0, class_idx].item()

        pred_class_name = idx_to_class[class_idx]

        print(Fore.GREEN + "预测类别：" + pred_class_name)
        print(Fore.GREEN + "预测置信度：{:.4f}".format(confidence))

        # 清空梯度
        self.model.zero_grad()

        # 取目标类别对应的分数
        target_score = output[0, class_idx]

        # 反向传播，计算目标类别对卷积特征图的梯度
        target_score.backward()

        # activations: [1, C, H, W]
        # gradients:   [1, C, H, W]
        activations = self.activations[0]
        gradients = self.gradients[0]

        # 对梯度在空间维度求平均，得到每个通道的重要性权重
        # weights: [C]
        weights = gradients.mean(dim=(1, 2))

        # 对每个通道的特征图加权求和
        cam = torch.zeros(
            activations.shape[1:],
            dtype=torch.float32
        ).to(device)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU：只保留对该类别有正向贡献的区域
        cam = torch.relu(cam)

        # 归一化到 0~1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = cam.cpu().numpy()

        return cam, pred_class_name, confidence

    def remove_hooks(self):
        """
        删除 hook，避免重复注册。
        """
        self.forward_hook.remove()
        self.backward_hook.remove()


# =========================
# 7. 热力图叠加到原图
# =========================

def overlay_cam_on_image(original_image, cam, save_path):
    """
    将 Grad-CAM 热力图叠加到原始图片上。
    """
    # 原图也 resize 到 256x256，保持和模型输入一致
    original_image = original_image.resize((256, 256))
    image_np = np.array(original_image).astype(np.float32) / 255.0

    # cam resize 到 256x256
    cam_image = Image.fromarray(np.uint8(cam * 255))
    cam_image = cam_image.resize((256, 256), resample=Image.BILINEAR)
    cam_np = np.array(cam_image).astype(np.float32) / 255.0

    # 使用 matplotlib 的 jet 色图生成热力图
    heatmap = plt.get_cmap("jet")(cam_np)[:, :, :3]

    # 叠加：原图 60%，热力图 40%
    overlay = 0.6 * image_np + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(cam_np, cmap="jet")
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print(Fore.GREEN + "Grad-CAM 结果已保存：" + save_path)


# =========================
# 8. 主程序
# =========================

if __name__ == "__main__":
    original_image, input_tensor = load_image(IMAGE_PATH)

    grad_cam = GradCAM(model, target_layer)

    cam, pred_class_name, confidence = grad_cam.generate(input_tensor)

    overlay_cam_on_image(
        original_image=original_image,
        cam=cam,
        save_path=SAVE_PATH
    )

    grad_cam.remove_hooks()