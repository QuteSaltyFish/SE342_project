# -*- coding: utf-8 -*-

'''
* 检测的过程还没有被写进去，请修改第48行，self.result为检测结果，self.img为原图，通道顺序是rgb
* 检测结果颜色填充，我在对话框里写的是"红-朝上，绿-朝下，蓝-立着"，如果不是这样在"dialogResult"里面第43行修改
* 默认的图片地址在第18行可以修改
'''

import sys
from mainWindow import *
from dialogResult import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QDialog
import cv2

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.path = "./123.png"
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
        self.pixMap = QtGui.QPixmap(self.path).scaledToHeight(480,1)
        self.label_2.resize(self.pixMap.width(), self.pixMap.height())
        self.label_2.setPixmap(self.pixMap)

    def imgRead(self):
        fname = QFileDialog.getOpenFileName()
        self.path = fname[0]
        self.img = cv2.cvtColor(cv2.imread(self.path), cv2.COLOR_BGR2RGB)
        self.setImg()

    # 点击
    def process(self):
        # TODO: ”self.result“是“检测的结果”，下面一行是需要修改的
        self.result = self.img
        dialogResult = DialogResult(self)
        dialogResult.show()


# 展示测试结果
# 请保证结果为rgb图像，不是bgr
class DialogResult(QDialog, Ui_Dialog):
    def __init__(self, parent=None):
        super(DialogResult,self).__init__(parent)
        self.setupUi(self)
        self.resultImg = parent.result
        self.getPixMap()
        self.labelResult.setPixmap(self.pixMap)

    def getPixMap(self, height=480):
        h, w, c = self.resultImg.shape
        qimg = QtGui.QImage(self.resultImg.data, w, h, self.resultImg.strides[0], QtGui.QImage.Format_RGB888)
        self.pixMap = QtGui.QPixmap.fromImage(qimg)
        self.pixMap = self.pixMap.scaledToHeight(480,1)
        print(self.pixMap.width(), self.pixMap.height())
        self.labelResult.resize(self.pixMap.width(), self.pixMap.height())


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    sys.exit(app.exec_())
