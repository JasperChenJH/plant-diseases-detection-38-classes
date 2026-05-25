# 文件路径
import os
import json
import time
from collections import Counter

import torch
import torchvision
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from colorama import Fore
from tqdm import tqdm

from src.model import PlainCNN


# =========================
# 0. 基础配置
# =========================

# 关闭 TensorFlow oneDNN 提示，主要是 tensorboard 可能触发
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# GPU / CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN + "当前设备：" + str(device))

# 数据集路径
ROOT_DIR = "../data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
VALID_DIR = os.path.join(ROOT_DIR, "valid")

# 输出目录
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练参数
batch_size = 32
epoch = 20
learning_rate = 1e-4

# 是否使用小样本过拟合测试
USE_SMALL_DATASET = True
SMALL_NUM = 200

# 固定随机种子，保证每次抽样一致
torch.manual_seed(0)


# =========================
# 1. 数据预处理
# =========================

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])

transform_valid = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])


# =========================
# 2. 读取数据集
# =========================

train_dataset_full = ImageFolder(TRAIN_DIR, transform=transform_train)
valid_dataset_full = ImageFolder(VALID_DIR, transform=transform_valid)

print(Fore.GREEN + "完整训练集类别数：" + str(len(train_dataset_full.classes)))
print(Fore.GREEN + "完整验证集类别数：" + str(len(valid_dataset_full.classes)))
print(Fore.GREEN + "完整训练集样本数：" + str(len(train_dataset_full)))
print(Fore.GREEN + "完整验证集样本数：" + str(len(valid_dataset_full)))


# =========================
# 3. 保存类别映射
# =========================

class_to_idx = train_dataset_full.class_to_idx
json_path = os.path.join(OUTPUT_DIR, "class_to_idx.json")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=4)

print(Fore.GREEN + "类别映射已保存：" + json_path)


# =========================
# 4. 打印每个类别数量
# =========================

train_class_count = Counter(train_dataset_full.targets)

idx_to_class = {
    idx: class_name
    for class_name, idx in class_to_idx.items()
}

print(Fore.GREEN + "\n训练集中每个类别的图像数量：")
for idx in range(len(idx_to_class)):
    class_name = idx_to_class[idx]
    count = train_class_count[idx]
    print(Fore.GREEN + f"{idx}: {class_name} -> {count} 张")


# =========================
# 5. 小样本过拟合测试
# =========================

if USE_SMALL_DATASET:
    print(Fore.YELLOW + f"\n当前使用小样本过拟合测试：{SMALL_NUM} 张")

    # 从完整训练集中随机抽取 200 张
    indices = torch.randperm(len(train_dataset_full))[:SMALL_NUM]

    # 训练集和验证集使用同一批图片
    # 目的：测试模型能不能记住这 200 张图片
    train_dataset = Subset(train_dataset_full, indices)
    valid_dataset = Subset(train_dataset_full, indices)

else:
    print(Fore.YELLOW + "\n当前使用完整训练集和验证集")

    train_dataset = train_dataset_full
    valid_dataset = valid_dataset_full


print(Fore.GREEN + "当前训练集样本数：" + str(len(train_dataset)))
print(Fore.GREEN + "当前验证集样本数：" + str(len(valid_dataset)))


# =========================
# 6. DataLoader
# =========================

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)


# =========================
# 7. 模型、损失函数、优化器
# =========================

model = PlainCNN(num_classes=len(train_dataset_full.classes))
model = model.to(device)

print(model)

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# =========================
# 8. TensorBoard
# =========================

writer = SummaryWriter(log_dir="./logs")


# =========================
# 9. 训练与验证
# =========================

train_step = 0
valid_step = 0
best_valid_acc = 0.0

for i in range(epoch):
    print(Fore.GREEN + "\n第 {} 轮训练开始...".format(i + 1))

    # =========================
    # 训练阶段
    # =========================

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    start_time = time.time()

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {i + 1}/{epoch} 训练中",
        ncols=120
    )

    for data in train_bar:
        img, label = data

        img = img.to(device)
        label = label.to(device)

        # 前向传播
        output = model(img)

        # 计算损失
        loss_result = loss_fn(output, label)

        # 梯度清零
        optimizer.zero_grad()

        # 反向传播
        loss_result.backward()

        # 更新参数
        optimizer.step()

        train_step += 1

        batch_size_now = label.size(0)

        # 累加训练 loss
        train_loss_sum += loss_result.item() * batch_size_now

        # 累加训练正确数
        train_correct += (output.argmax(1) == label).sum().item()

        # 累加训练样本数
        train_total += batch_size_now

        current_train_loss = train_loss_sum / train_total
        current_train_acc = train_correct / train_total

        train_bar.set_postfix(
            loss="{:.4f}".format(current_train_loss),
            acc="{:.4f}".format(current_train_acc),
            step=train_step
        )

        writer.add_scalar("train_loss_step", loss_result.item(), train_step)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    end_time = time.time()

    print(
        Fore.GREEN +
        "第{}轮训练结束，平均损失={:.4f}，训练准确率={:.4f}，耗时={:.2f}秒".format(
            i + 1,
            train_loss,
            train_acc,
            end_time - start_time
        )
    )

    writer.add_scalar("epoch_train_loss", train_loss, i + 1)
    writer.add_scalar("epoch_train_acc", train_acc, i + 1)


    # =========================
    # 验证阶段
    # =========================

    print(Fore.GREEN + "第{}轮验证开始...".format(i + 1))

    model.eval()

    valid_loss_sum = 0.0
    valid_correct = 0
    valid_total = 0

    valid_bar = tqdm(
        valid_loader,
        desc=f"Epoch {i + 1}/{epoch} 验证中",
        ncols=120
    )

    with torch.no_grad():
        for data in valid_bar:
            img, label = data

            img = img.to(device)
            label = label.to(device)

            output = model(img)

            loss_result = loss_fn(output, label)

            batch_size_now = label.size(0)

            valid_loss_sum += loss_result.item() * batch_size_now
            valid_correct += (output.argmax(1) == label).sum().item()
            valid_total += batch_size_now

            current_valid_loss = valid_loss_sum / valid_total
            current_valid_acc = valid_correct / valid_total

            valid_bar.set_postfix(
                loss="{:.4f}".format(current_valid_loss),
                acc="{:.4f}".format(current_valid_acc)
            )

    valid_step += 1

    valid_loss = valid_loss_sum / valid_total
    valid_acc = valid_correct / valid_total

    print(
        Fore.GREEN +
        "第{}次验证，损失值={:.4f}，准确率={:.4f}".format(
            valid_step,
            valid_loss,
            valid_acc
        )
    )

    writer.add_scalar("epoch_valid_loss", valid_loss, i + 1)
    writer.add_scalar("epoch_valid_acc", valid_acc, i + 1)


    # =========================
    # 保存模型
    # =========================

    # 保存每一轮模型
    torch.save(
        model,
        os.path.join(OUTPUT_DIR, "model_{}.pth".format(i + 1))
    )

    # 保存最佳模型
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc

        best_path = os.path.join(OUTPUT_DIR, "best_plain_cnn.pth")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "valid_acc": best_valid_acc,
                "epoch": i + 1
            },
            best_path
        )

        print(
            Fore.GREEN +
            "保存当前最佳模型：{}，valid_acc={:.4f}".format(
                best_path,
                best_valid_acc
            )
        )


writer.close()

print(Fore.GREEN + "\n训练结束！")
print(Fore.GREEN + "最佳验证准确率：{:.4f}".format(best_valid_acc))