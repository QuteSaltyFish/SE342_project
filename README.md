# SE342_project
This is the final project of SE342.
The main problem is to detect the different posture of the bottle cap.


# How to run our code
`python main.py`
# The Q&A Part

Q:看一下数据集

A:这是我们的数据集，数据比较一般，瓶盖的阴影比较大，也有许多在边界上的竖着的瓶盖，所以在尝试使用境地那算法的时候并不能有很好的表现，于是我们选择了使用深度学习。对数据进行了大量的数据增强，比如裁剪，能够很好的去处边界的数据的影响，但是最后的输出结果还是有非常多的噪声，所以我们继续用经典方法进行了后处理，让结果更加好看。

Q:是直接在训练集上面跑的吗

不是，我们刚刚的那个数据图是对整个数据做了5fold 的cross-validation做出来的，最后平均每一个fold的结果画出来的曲线，实际上训练的时候是分成2个fold，每一个fold用于测试另外一半的数据。