import torch as t
import torchvision as tv

import numpy.random as np

# 给定原始图像img和想要获取的图像维度[size_h,size_w]，求出图中的红色矩形大小；想要获取的图像的左上角坐标一定落在红色矩阵中


def randomCrop(img, label, size_h, size_w):
    rows, cols = img.shape[:2]
    left_h = np.randint(0, rows-size_h)
    left_w = np.randint(0, cols-size_w)
    crop_img = img[left_h:left_h+size_h, left_w:left_w+size_w]
    crop_label = label[left_h:left_h+size_h, left_w:left_w+size_w]
    return crop_img, crop_label

# 水平翻转


def horizontalFlip(img, label):
    return flip(img, -1), flip(label, -1)

# 竖直翻转


def verticalFlip(img):
    return flip(img, -2), flip(label, -2)


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    inds = tuple(slice(None, None) if i != dim
                 else x.new(t.arange(x.size(i)-1, -1, -1).tolist()).long()
                 for i in range(x.dim()))
    return x[inds]


def transform(self, image, mask):
    # 拿到角度的随机数。angle是一个-180到180之间的一个数
    angle = transforms.RandomRotation.get_params([-180, 180])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image = tf.rotate(image, angle, resample=Image.NEAREST)
    mask = tf.rotate(mask, angle, resample=Image.NEAREST)
    # 自己写随机部分，50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)
    # 也可以实现一些复杂的操作
    # 50%的概率对图像放大后裁剪固定大小
    # 50%的概率对图像缩小后周边补0，并维持固定大小
    if random.random() > 0.5:
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.25, 1.0), ratio=(1, 1))
        image = tf.resized_crop(image, i, j, h, w, 256)
        mask = tf.resized_crop(mask, i, j, h, w, 256)
    else:
        pad = random.randint(0, 192)
        image = tf.pad(image, pad)
        image = tf.resize(image, 256)
        mask = tf.pad(mask, pad)
        mask = tf.resize(mask, 256)
    # 转换为tensor并做归一化
    image = tf.to_tensor(image)
    image = tf.normalize(image, [0.5], [0.5])
    mask = tf.to_tensor(mask)
    mask = tf.normalize(mask, [0.5], [0.5])
    return image, mask


if __name__ == "__main__":
    a = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(a)
    print(flip(a, -2))
    print(t.flip(a[..., ::-1]))
