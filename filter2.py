import cv2
import sys
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMessageBox

import UI.main
import os
import numpy as np

class ExampleApp(QtWidgets.QMainWindow, UI.main.Ui_MainWindow):

    base_image = ""
    image_to_filter = ""
    sigma_color = 0
    sigma_space = 0
    x1 = -1
    x2 = -1
    y1 = -1
    y2 = -1
    def __init__(self):

        super().__init__()
        self.setupUi(self)

        self.Start.setDisabled(1)
        self.sigma_color_slider.setDisabled(1)
        self.sigma_space_slider.setDisabled(1)
        self.Crop.setDisabled(1)
        self.Save.setDisabled(1)

        self.OpenImage.clicked.connect(self.openFileNameDialog)

        self.Start.clicked.connect(self.filter)
        self.Save.clicked.connect(self.save_image)
        self.Crop.clicked.connect(self.crop_image)
        self.Cancel.clicked.connect(self.cancel_crop)

        self.InputImage.setStyleSheet("border: 1px solid black;")
        self.InputImage.mousePressEvent=self.get_pixel

        self.OutputImage.setStyleSheet("border: 1px solid black;")

        self.sigma_color_slider.setMinimum(0)
        self.sigma_color_slider.setMaximum(255)
        self.sigma_color_slider.setSingleStep(1)
        self.sigma_color_slider.valueChanged.connect(self.set_sigma)

        self.sigma_space_slider.setMinimum(0)
        self.sigma_space_slider.setMaximum(255)
        self.sigma_space_slider.setSingleStep(1)
        self.sigma_space_slider.valueChanged.connect(self.set_space)




    # Отмена обрезкии
    def cancel_crop(self):
        self.image_to_filter = self.base_image
        pixmap = QPixmap(self.base_image)
        self.InputImage.setPixmap(pixmap)

    # получение координат по клику мыши
    def get_pixel(self, event):
        x = event.pos().x()
        y = event.pos().y()

        if(self.x1 == -1):
            self.x1 = x
            self.y1 = y
        elif(self.x2 == -1):
            self.x2 = x
            self.y2 = y
            self.Crop.setDisabled(0)
        else:
            self.x1 = x
            self.y1 = y
            self.x2 = -1
            self.y2 = -1
            self.Crop.setDisabled(1)


        var = "Координаты: ("+str(self.x1)+ ","+str(self.y1)+ ") - ("+str(self.x2)+ ","+str(self.y2)+ ")"

        self.Coords.setText(var)
        print("(",self.x1,",",self.y1, ") - (", self.x2,",",self.y2,")")

    def crop_image(self):
        temp_path = "temp.png"

        input_image = cv2.imread(self.image_to_filter )

        if(self.y1 > self.y2):
            self.y1,self.y2 = self.y2,self.y1
        if(self.x1 > self.x2):
            self.x1,self.x2 = self.x2,self.x1

        image = input_image[self.y1:self.y2, self.x1:self.x2]
        cv2.imwrite(temp_path, image)
        self.image_to_filter=temp_path
        input_image = cv2.imread(temp_path)

        height, width, channel = input_image.shape

        bytes_per_line = channel * width
        print(input_image.shape, bytes_per_line)
        qImg = QImage(input_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap(qImg)

        self.InputImage.setPixmap(pixmap)


    # установка параметров
    def set_sigma(self):
        self.sigma_color_number.display(self.sigma_color_slider.value())
        self.sigma_color=self.sigma_color_slider.value()
    def set_space(self):
        self.sigma_space_number.display(self.sigma_space_slider.value())
        self.sigma_space=self.sigma_space_slider.value()


    # Сохранение картинки
    def save_image(self):
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Сохранение изображения","","Images (*.png)")
        pixmap = self.OutputImage.pixmap()
        pixmap.save(fileName, 'png')

    # Диалог открытия картинки
    def openFileNameDialog(self):
        self.base_image= ""

        self.base_image, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Images (*.png;*.jpeg)")
        self.image_to_filter=self.base_image
        if(self.base_image!= ""):
            pixmap = QPixmap(self.base_image)
            self.InputImage.setPixmap(pixmap)
            self.Start.setDisabled(0)

            self.sigma_color_slider.setDisabled(0)
            self.sigma_space_slider.setDisabled(0)
            self.OutputImage.clear()
            self.Save.setDisabled(1)
        else:
            self.OutputImage.clear()
            self.InputImage.clear()
            self.Start.setDisabled(1)
            self.sigma_color_slider.setDisabled(1)
            self.sigma_space_slider.setDisabled(1)
            self.Crop.setDisabled(1)

    # Фильтрация
    def filter(self):

        self.Start.setDisabled(1)
        self.Save.setDisabled(1)


        input_image = cv2.imread(self.image_to_filter)
        image = cv2.bilateralFilter(input_image, self.sigma_color, self.sigma_space, cv2.BORDER_REFLECT)

        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg  = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap(qImg)

        self.OutputImage.setPixmap(pixmap)

        self.Save.setDisabled(0)
        self.Start.setDisabled(0)
        QMessageBox.about(self, "Инфо", "Фильтрация завершена")


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp()
    window.show()
    app.exec_()




main()


