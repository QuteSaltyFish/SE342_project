# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialogResult.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        #Dialog.resize(707, 544)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setStyleSheet("font: 13pt \"Arial\";")
        self.label.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label.setTextFormat(QtCore.Qt.AutoText)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.labelResult = QtWidgets.QLabel(Dialog)
        self.labelResult.setAlignment(QtCore.Qt.AlignCenter)
        self.labelResult.setObjectName("labelResult")
        self.gridLayout.addWidget(self.labelResult, 0, 0, 1, 1)
        self.okCancelBtn = QtWidgets.QDialogButtonBox(Dialog)
        self.okCancelBtn.setOrientation(QtCore.Qt.Horizontal)
        self.okCancelBtn.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.okCancelBtn.setCenterButtons(False)
        self.okCancelBtn.setObjectName("okCancelBtn")
        self.gridLayout.addWidget(self.okCancelBtn, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.okCancelBtn.accepted.connect(Dialog.accept)
        self.okCancelBtn.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Result"))
        self.label.setText(_translate("Dialog", "R:UP\n"
"G:EDGE\n"
"B:DOWN"))
        self.labelResult.setText(_translate("Dialog", "TextLabel"))
