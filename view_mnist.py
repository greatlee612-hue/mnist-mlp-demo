"""读取 MNIST 的 .ubyte 文件并画出来。用法: python view_mnist.py"""
import struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

RAW = r"mnist_data/MNIST/raw"
IMGS = f"{RAW}/t10k-images-idx3-ubyte"
LBLS = f"{RAW}/t10k-labels-idx1-ubyte"


def read_images(path):
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 0x00000803, f"不是 IDX3 图像文件: {magic:#x}"
        data = np.frombuffer(f.read(n * rows * cols), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def read_labels(path):
    with open(path, 'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        assert magic == 0x00000801, f"不是 IDX1 标签文件: {magic:#x}"
        data = np.frombuffer(f.read(n), dtype=np.uint8)
    return data


def main():
    imgs = read_images(IMGS)
    lbls = read_labels(LBLS)
    print(f"读入 {imgs.shape[0]} 张图, 尺寸 {imgs.shape[1]}x{imgs.shape[2]}")
    print(f"读入 {lbls.shape[0]} 个标签, 前10个: {list(lbls[:10])}")
    print(f"像素值范围: {imgs.min()} ~ {imgs.max()}")

    # 拼一张 8x8 的缩略图墙
    fig, axes = plt.subplots(8, 8, figsize=(9, 9))
    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i], cmap='gray')
        ax.set_title(f"#{i} → {lbls[i]}", fontsize=9)
        ax.axis('off')
    fig.suptitle("MNIST 测试集前 64 张 (标题 = 编号 → 真实标签)", fontsize=13)
    fig.tight_layout()
    fig.savefig("mnist_preview.png", dpi=110)
    print("已保存 -> mnist_preview.png")

    # 放一张大图，看清像素
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(imgs[0], cmap='gray')
    ax2.set_title(f"第 0 张，真实标签 = {lbls[0]}，灰度值 0(黑)~255(白)")
    ax2.axis('off')
    fig2.tight_layout()
    fig2.savefig("mnist_single.png", dpi=110)
    print("已保存 -> mnist_single.png")

    plt.show()   # 弹窗显示，关掉窗口即结束


if __name__ == '__main__':
    main()
