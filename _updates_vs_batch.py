"""决定性实验：把「更新次数」和「batch 大小」分离开。

如果大 batch 的问题只是更新次数少，那么：
在相同的更新次数下，不同 batch 的准确率应该接近，
甚至大 batch 更好（因为梯度更准、每次更新更有效）。
"""
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

ROOT = 'C:/Users/yushengliao/Desktop/mnistdemo/mnist_data/'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        return torch.nn.functional.log_softmax(self.fc4(x), dim=1)


def make_loader(train, bs):
    tf = transforms.Compose([transforms.ToTensor()])
    return DataLoader(MNIST(root=ROOT, train=train, download=False, transform=tf),
                      batch_size=bs, shuffle=True, pin_memory=True)


def evaluate(net, test):
    net.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in test:
            x, y = x.to(DEV), y.to(DEV)
            out = net(x.view(-1, 784))
            c += (out.argmax(1) == y).sum().item()
            t += y.size(0)
    return c / t


def train_curve(bs, checkpoints, lr=0.001, seed=0):
    """训练到 max(checkpoints) 次更新，在指定的更新次数上记录测试准确率。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tr = make_loader(True, bs)
    te = make_loader(False, bs)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    steps_per_epoch = len(tr)
    maxu = max(checkpoints)
    got, u = {}, 0

    while True:
        net.train()
        for x, y in tr:
            x, y = x.to(DEV), y.to(DEV)
            opt.zero_grad()
            loss = torch.nn.functional.nll_loss(net(x.view(-1, 784)), y)
            loss.backward()
            opt.step()
            u += 1
            if u in checkpoints:
                got[u] = evaluate(net, te)
            if u >= maxu:
                return got, steps_per_epoch


CHECKPOINTS = [45, 150, 500, 1500, 3750]

CASES = [
    (16,    CHECKPOINTS),
    (64,    [c for c in CHECKPOINTS if c <= 3750]),
    (256,   [c for c in CHECKPOINTS if c <= 1500]),
    (4096,  [c for c in CHECKPOINTS if c <= 500]),
]

print(f'device = {DEV}   lr = 0.001 (固定，不缩放)')
print('目标：比较「相同的更新次数」下不同 batch 的表现\n')

results = {}
for bs, cps in CASES:
    spe = 60000 // bs
    print(f'--- batch={bs}   ({spe} 步/epoch) ---', flush=True)
    got, spe = train_curve(bs, cps)
    results[bs] = got
    for c in sorted(got):
        ep = c / spe
        print(f'    更新 {c:>6d} 次  (≈{ep:6.2f} epoch)   准确率 {got[c]*100:6.2f}%', flush=True)

print()
print('=' * 70)
print('准确率 vs 更新次数（固定 lr=0.001）')
print('=' * 70)
hdr = f'{"batch":>7s} |' + ''.join(f'{c:>12d}' for c in CHECKPOINTS)
print(hdr)
print('-' * len(hdr))
for bs, _ in CASES:
    row = f'{bs:>7d} |'
    for c in CHECKPOINTS:
        v = results[bs].get(c)
        row += f'{v*100:11.2f}%' if v is not None else f'{"--":>12s}'
    print(row)
