from torch import nn
import torch


class PlainCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(PlainCNN, self).__init__()

        self.module = nn.Sequential(
            # 输入: 3 x 256 x 256

            # 第一组卷积：提取低级特征，如边缘、颜色变化、简单纹理
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            # 输出: 32 x 85 x 85

            # 第二组卷积：提取更复杂的局部纹理和病斑边缘
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            # 输出: 64 x 28 x 28

            # 第三组卷积：提取病斑形状、叶片纹理等更高级特征
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
            # 输出: 128 x 9 x 9

            # 第四组卷积：进一步提取高级病害语义特征
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 输出: 256 x 9 x 9

            # 第五组卷积：使用 5x5 卷积扩大感受野
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 输出: 512 x 9 x 9

            # 展平
            nn.Flatten(),

            # 全连接分类部分
            nn.Linear(512 * 9 * 9, 1568),
            nn.ReLU(inplace=True),

            # Dropout 防止过拟合
            nn.Dropout(p=0.5),

            # 输出 38 类
            nn.Linear(1568, num_classes)
        )

    def forward(self, x):
        x = self.module(x)
        return x


if __name__ == '__main__':
    model = PlainCNN(num_classes=38)

    # 测试输入：batch_size=8，RGB三通道，图片大小256x256
    test = torch.ones(8, 3, 256, 256)

    res = model(test)

    print(res.shape)