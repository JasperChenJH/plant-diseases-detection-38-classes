# 文件路径
import json
from tqdm import tqdm
import os
import time
from collections import Counter
import torch
import torchvision
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from src.model import PlainCNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Gpu训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(Fore.GREEN + "当前设备：" + str(device))
# 数据集路径
ROOT_DIR = "../data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
TEST_DIR = "../data/test/test"
TRAIN_DIR = os.path.join(ROOT_DIR, "train")
VALID_DIR = os.path.join(ROOT_DIR, "valid")
# 输出目录：保存模型、类别映射、训练曲线
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 创建dataset
transform_train = torchvision.transforms.Compose([
    # 缩放到 256 x 256
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
])
transform_valid = torchvision.transforms.Compose([
    # 缩放到 256 x 256
    torchvision.transforms.Resize((256, 256)),
    # 转换为张量ToTensor
    torchvision.transforms.ToTensor(),
])
valid_dataset = ImageFolder(VALID_DIR, transform=transform_valid)
train_dataset = ImageFolder(TRAIN_DIR, transform=transform_train)
print(Fore.GREEN+"训练集类别数：", len(train_dataset.classes))
print(Fore.GREEN+"验证集类别数：", len(valid_dataset.classes))
print(Fore.GREEN+"训练集样本数：", len(train_dataset))
print(Fore.GREEN+"验证集样本数：", len(valid_dataset))
# 创建映射json文件
class_to_idx = train_dataset.class_to_idx
json_path = os.path.join(OUTPUT_DIR, "class_to_idx.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(class_to_idx, f, ensure_ascii=False, indent=4)
# 打印每个类别的数量
train_class_count = Counter(train_dataset.targets)
print(Fore.GREEN+"训练集类别统计："+ str(train_class_count))
# 转成：编号 -> 类别名
idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}
print(Fore.GREEN + "\n训练集中每个类别的图像数量：")
for idx in range(len(idx_to_class)):
    class_name = idx_to_class[idx]
    count = train_class_count[idx]
    print(Fore.GREEN + f"{idx}: {class_name} -> {count} 张")
# 构建DataLoader
batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# 搭建模型
model = PlainCNN(num_classes=len(train_dataset.classes))
model = model.to(device)
print(model)
# 损失函数
loss = torch.nn.CrossEntropyLoss()
loss = loss.to(device)
# 优化器
learning_rate = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#步骤参数
epoch = 10
train_step = 0
valid_step = 0
writer = SummaryWriter(log_dir="./logs")
for i in range(epoch):
    print(Fore.GREEN + "第 {} 轮训练开始...".format(i + 1))
    # 训练模式
    model.train()
    start_time = time.time()
    for data in train_loader:
        # 数据读取
        img, label = data
        img = img.to(device)
        label = label.to(device)
        # 前向传播
        output = model(img)
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失
        loss_result = loss(output, label)
        # 反向传播
        loss_result.backward()
        # 优化器更新参数
        optimizer.step()
        train_step += 1
        if train_step % 100 == 0:
            end_time = time.time()
            print("训练时间:{}".format(end_time-start_time))
            start_time = time.time()
            print("第{}次训练，损失值为{}".format(train_step, loss_result.item()))
            writer.add_scalar("train_loss", loss_result.item(), train_step)
    # 验证模式
    global_valid_loss = 0
    global_accuracy = 0
    valid_total = 0
    print(Fore.GREEN+"第{}轮验证开始...".format(i + 1))
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            # 数据读取
            img, label = data
            img = img.to(device)
            # 验证数据总量
            valid_total += label.shape[0]
            label = label.to(device)
            # 前向传播
            output = model(img)
            # 计算损失
            loss_result = loss(output, label)
            # 损失求和
            global_valid_loss += loss_result.item() * label.size(0)
            # 准确率
            accuracy = (output.argmax(1) == label).sum().item()
            # 准确率求和
            global_accuracy += accuracy
        valid_step += 1
        print(Fore.GREEN+"第{}次验证，损失值为{}，准确率为{}".format(valid_step, global_valid_loss/valid_total, global_accuracy/valid_total))
        writer.add_scalar("test_loss", global_valid_loss/valid_total, valid_step)
        writer.add_scalar("test_accuracy", global_accuracy/valid_total, valid_step)
    torch.save(model, "model_{}.pth".format(i))
writer.close()
