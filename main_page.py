import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 이미지 업로드 버튼 생성
        self.img_upload_btn = QPushButton("이미지 업로드")
        self.img_upload_btn.clicked.connect(self.img_upload)

        # box grid 생성
        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget()
        self.hbox.addStretch(1)

        self.vbox = QVBoxLayout()
        self.vbox.addStretch(1)
        self.vbox.addLayout(self.hbox)
        self.vbox.addStretch(1)

        self.widget = QWidget()
        self.widget.setLayout(self.vbox)
        self.centralWidget(self.widget)


        self.setWindowTitle("recycle_project")
        # 모니터를 x, y축으로 구분, (모니터의 왼쪽 상단은 (0, 0))
        # x, y, 창의 넓이, 창의 높이 (좌표의 기준은 창의 왼쪽 상단)
        self.setGeometry(300, 100, 1500, 900)
        self.show()

    def img_upload(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())