import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib
matplotlib.use('TkAgg')
# matplotlib图表绘制工具
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 有 GPU 就用 GPU，没有就退回 CPU

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)  #这是第一个全连接层（Fully Connected Layer），它将输入的特征维度从 28 * 28（通常是28x28像素的图像展平后的长度）转换为 64 个输出单元
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)#这是第四个全连接层，将 64 映射到 10 ，10 对应 0-9 的数字

    def forward(self, x):
        x=torch.nn.functional.relu(self.fc1(x))

        x=torch.nn.functional.relu(self.fc2(x))
        x=torch.nn.functional.relu(self.fc3(x))
        x=torch.nn.functional.log_softmax(self.fc4(x), dim=1)#做归一化求概率最大值
        return x

def get_data_loader(is_train):
    to_tensor=transforms.Compose([transforms.ToTensor()])
    data_set=MNIST(root='mnist_data/', train=is_train, download=True, transform=to_tensor)
    return DataLoader(data_set, batch_size=128, shuffle=True, pin_memory=True)

def evaluate(test_data,net):
    net.eval()  # 将模型设置为评估模式
    n_correct=0
    n_total=0
    with torch.no_grad():#禁用梯度计算。在评估模型时，我们不需要计算梯度，因为不会进行反向传播，防止题目泄露
        for (x,y) in test_data:
            #x是图片像素(28*28) y是标准答案
            x, y = x.to(device), y.to(device)  # 移动数据到设备
            outputs = net.forward(x.view(-1,28*28))
            # 整个 batch 一起比对，比逐个样本循环快几十倍
            n_correct += (torch.argmax(outputs, dim=1) == y).sum().item()
            # 判断一行中概率最大的数字对应的数字是否等于答案
            n_total += y.size(0)
    return n_correct / n_total

def main():
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    net.to(device)  # 将模型移动到选择的设备

    print("initial accuracy:",evaluate(test_data,net))
    optimizer =torch.optim.Adam(net.parameters(),lr=0.001)

    for epoch in range(10):
        net.train()  # 将模型设置为训练模式
        for(x,y) in train_data:
            x, y = x.to(device), y.to(device)  # 移动数据到设备
            net.zero_grad()#初始化
            outputs = net.forward(x.view(-1,28*28))#正向传播
            loss=torch.nn.functional.nll_loss(outputs,y)#计算差值
            loss.backward()#反向误差传播
            optimizer.step()#优化网络参数
        print("epoch:",epoch,"accuracy:",evaluate(test_data,net))

    plt.figure(figsize=(10, 5))
    net.eval()
    with torch.no_grad():
        for n, (x, y) in enumerate(test_data):
            if n > 3:
                break
            x = x.to(device)
            batch_out = net(x.view(-1, 28 * 28))
            pred = torch.argmax(batch_out, dim=1)
            plt.subplot(2, 2, n + 1)
            plt.imshow(x[0].view(28, 28).cpu().numpy(), cmap='gray')  # 确保在绘图前数据在CPU上
            plt.title(f"Pred:{pred[0].item()}, True:{y[0].item()}")
            plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

    # 学习率没变 batch大无噪声容易走进局部最优点 鲁棒性差泛化能力不好