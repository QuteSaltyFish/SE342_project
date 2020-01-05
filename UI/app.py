# -*- coding: utf-8 -*-

'''
* 检测的过程还没有被写进去，请修改第48行，self.result为检测结果，self.img为原图，通道顺序是rgb
* 检测结果颜色填充，我在对话框里写的是"红-朝上，绿-朝下，蓝-立着"，如果不是这样在"dialogResult"里面第43行修改
* 默认的图片地址在第18行可以修改
'''

from get_output.utils import img_label
from model.func import output
from model.unet_model import UNet
from PIL import Image
from model import dataloader
import sys
from UI.mainWindow import *
from UI.dialogResult import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog
import cv2

import json
import os
import numpy as np
import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

config = json.load(open('config.json'))
projPath = config["project_path"]
sys.path.append(projPath)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.path = "data/image/img1.jpg"
        self.img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        self.result = None
        self.setupUi(self)
        self.setupAction()
        self.setImg()

    def setupAction(self):
        self.detectBtn.clicked.connect(self.process)
        self.newImgBtn.clicked.connect(self.imgRead)
        self.actionclose.triggered.connect(self.close)
        self.actionopen.triggered.connect(self.imgRead)

    def setImg(self):
        self.pixMap = QtGui.QPixmap(self.path).scaledToHeight(480, 1)
        self.label_2.resize(self.pixMap.width(), self.pixMap.height())
        self.label_2.setPixmap(self.pixMap)

    def imgRead(self):
        fname = QFileDialog.getOpenFileName(
            directory='/home/wangmingke/Desktop/pictopick')
        if fname[0] == '':
            pass
        else:
            self.path = fname[0]
            self.img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
            self.setImg()

    # 点击
    def process(self):
        # TODO: ”self.result“是“检测的结果”，下面一行是需要修改的
        self.result = (output(self.path)*255).astype(np.uint8)
        self.result = cv2.resize(
            self.result, (self.result.shape[1]*3, self.result.shape[0]*3))
        self.closeopen_demor()
        self.result = img_label(self.img, cv2.resize(
            self.result, self.img.shape[1::-1]))
        self.img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        dialogResult = DialogResult(self)
        dialogResult.show()

    def testAll(self):
        data_root = config["Taining_Dir"]
        names = np.array(os.listdir(data_root))
        for i in range(len(names)):
            print(os.path.join(data_root, names[i]))
            self.img = cv2.cvtColor(cv2.imread(
                os.path.join(data_root, names[i])), cv2.COLOR_BGR2RGB)
            self.result = (
                output(os.path.join(data_root, names[i]))*255).astype(np.uint8)
            self.result = cv2.resize(
                self.result, (self.result.shape[1]*3, self.result.shape[0]*3)) 
            self.closeopen_demor()
            self.result = img_label(cv2.resize(
                self.img, self.result.shape[1::-1]), self.result)
            self.img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
            name = names[i].split('.')[0]
            cv2.imwrite(
                "final_output/img"+name+".png", cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR))
        self.close()

    # 消除噪点
    def closeopen_demor(self):
        r, g, b = cv2.split(self.result)
        thres = 15
        thresRED = 25

        gray = r
        ret, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (thresRED, thresRED))
        binaryr = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        r2 = cv2.morphologyEx(binaryr, cv2.MORPH_OPEN, kernel)

        gray = b
        ret, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (thresRED, thresRED))
        binaryb = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        b2 = cv2.morphologyEx(binaryb, cv2.MORPH_OPEN, kernel)

        gray = g
        ret, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (thres, thres))
        binaryg = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        g2 = cv2.morphologyEx(binaryg, cv2.MORPH_CLOSE, kernel)

        self.result = cv2.merge([r2, g2, b2])


# 展示测试结果
# 请保证结果为rgb图像，不是bgr
class DialogResult(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(DialogResult, self).__init__(parent)
        self.setupUi(self)
        self.resultImg = parent.result
        self.img = parent.img
        self.getPixMap()
        self.labelResult.setPixmap(self.pixMap)

    def getPixMap(self, height=480):
        h, w, c = self.resultImg.shape
        qimg = QtGui.QImage(self.resultImg.data, w, h,
                            self.resultImg.strides[0], QtGui.QImage.Format_RGB888)
        self.pixMap = QtGui.QPixmap.fromImage(qimg)
        self.pixMap = self.pixMap.scaledToHeight(600, 1)
        self.labelResult.resize(self.pixMap.width(), self.pixMap.height())


def main():
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())


def test():
    app = QApplication(sys.argv)
    myWin = MainWindow()
    # myWin.show()
    myWin.testAll()
    # sys.exit(app.exec_())


if __name__ == '__main__':
    main()
