'''
这个test的,就是先把代码的训练搞起来,找到一点头绪,然后就去切换我的PMCNN,争取搞一点噱头出来啊你说是不是
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.models as models

train_path = 'data/fer2013images/train/'
vaild_path = 'data/fer2013images/valid/'
data_path = 'data/fer2013/fer2013.csv'
emotion_dict = {'0': 'anger', '1': 'disgust', '2': 'fear', '3': 'happy', '4': 'sad', '5': 'surprised', '6': 'normal'}


def make_dir():
    for i in range(0, 7):
        p1 = os.path.join(train_path, str(i))
        p2 = os.path.join(vaild_path, str(i))
        if not os.path.exists(p1):
            os.makedirs(p1)
        if not os.path.exists(p2):
            os.makedirs(p2)


def save_images():
    df = pd.read_csv(data_path)
    # 那他这样写还真没错,用了两个7个元素的数组来分别存一下两边的图片的编号,这真的可以的
    t_i = [1 for i in range(0, 7)]
    v_i = [1 for i in range(0, 7)]
    for index in range(len(df)):
        emotion = df.loc[index][0]
        image = df.loc[index][1]
        usage = df.loc[index][2]
        data_array = list(map(float, image.split()))
        data_array = np.asarray(data_array)
        image = data_array.reshape(48, 48)
        im = Image.fromarray(image).convert('L')  # 8位黑白图片
        if (usage == 'Training'):
            t_p = os.path.join(train_path, str(emotion), '{}.jpg'.format(t_i[emotion]))
            im.save(t_p)
            t_i[emotion] += 1
            # print(t_p)
        else:
            v_p = os.path.join(vaild_path, str(emotion), '{}.jpg'.format(v_i[emotion]))
            im.save(v_p)
            v_i[emotion] += 1
            # print(v_p)



BATCH_SIZE = 128
LR = 0.01
EPOCH = 60
DEVICE = torch.device('cuda')

path_train = 'data/fer2013images/train'
path_vaild = 'data/fer2013images/valid'

transforms_train = transforms.Compose([
    transforms.Grayscale(),  # 使用ImageFolder默认扩展为三通道，重新变回去就行
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度和对比度
    transforms.ToTensor()
])
transforms_vaild = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

data_train = torchvision.datasets.ImageFolder(root=path_train, transform=transforms_train)
data_vaild = torchvision.datasets.ImageFolder(root=path_vaild, transform=transforms_vaild)

train_set = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
vaild_set = torch.utils.data.DataLoader(dataset=data_vaild, batch_size=BATCH_SIZE, shuffle=False)

train_loss = []
train_ac = []
vaild_loss = []
vaild_ac = []



class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 残差神经网络
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels  # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


resnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 2))
resnet.add_module("resnet_block3", resnet_block(128, 256, 2))
resnet.add_module("resnet_block4", resnet_block(256, 512, 2))
resnet.add_module("global_avg_pool", GlobalAvgPool2d())  # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(Reshape(), nn.Linear(512, 7)))


resnet18 = models.resnet18()

model = resnet
model.to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
# optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print(model)

train_loss = []
train_ac = []
vaild_loss = []
vaild_ac = []
y_pred = []


def train(model, device, dataset, optimizer, epoch):
    model.train()
    correct = 0
    for i, (x, y) in tqdm(enumerate(dataset)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

    train_ac.append(correct / len(data_train))
    train_loss.append(loss.item())
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch, loss, correct, len(data_train), 100 * correct / len(data_train)))


def vaild(model, device, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            pred = output.max(1, keepdim=True)[1]
            global y_pred
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()

    vaild_ac.append(correct / len(data_vaild))
    vaild_loss.append(loss.item())
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss, correct, len(data_vaild), 100. * correct / len(data_vaild)))


def RUN():
    for epoch in range(1, EPOCH + 1):
        '''if epoch==15 :
            LR = 0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        if(epoch>30 and epoch%15==0):
            LR*=0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        '''
        # 尝试动态学习率
        train(model, device=DEVICE, dataset=train_set, optimizer=optimizer, epoch=epoch)
        vaild(model, device=DEVICE, dataset=vaild_set)
        torch.save(model, 'm0.pth')




# vaild(model,device=DEVICE,dataset=vaild_set)

def print_plot(train_plot, vaild_plot, train_text, vaild_text, ac, name):
    x = [i for i in range(1, len(train_plot) + 1)]
    plt.plot(x, train_plot, label=train_text)
    plt.plot(x[-1], train_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (train_plot[-1] * 100) if ac else "%.4f" % (train_plot[-1]), xy=(x[-1], train_plot[-1]))
    plt.plot(x, vaild_plot, label=vaild_text)
    plt.plot(x[-1], vaild_plot[-1], marker='o')
    plt.annotate("%.2f%%" % (vaild_plot[-1] * 100) if ac else "%.4f" % (vaild_plot[-1]), xy=(x[-1], vaild_plot[-1]))
    plt.legend()
    plt.savefig(name)










if __name__ == '__main__':
    # make_dir()
    # save_images()

    RUN()

    print_plot(train_loss, vaild_loss, "train_loss", "vaild_loss", False, "loss.jpg")
    print_plot(train_ac, vaild_ac, "train_ac", "vaild_ac", True, "ac.jpg")

    emotion = ["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"]
    sns.set()
    f, ax = plt.subplots()
    y_true = [emotion[i] for _, i in data_vaild]
    y_pred = [emotion[i] for i in y_pred]
    C2 = confusion_matrix(y_true, y_pred, labels=["angry", "disgust", "fear", "happy", "sad", "surprised", "neutral"])  # [0, 1, 2,3,4,5,6])
    # print(C2) #打印出来看看
    sns.heatmap(C2, annot=True, fmt='.20g', ax=ax)  # 热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.savefig('matrix.jpg')

