import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QWidget
import cv2

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # 이미지 박스
        self.image_box = QLabel("이미지 공간")

        # 이미지 업로드 버튼 생성
        self.img_upload_btn = QPushButton("이미지 업로드")
        self.img_upload_btn.clicked.connect(self.img_upload)

        # 이미지 분석 결과 텍스트
        self.result_text = QLabel("예시")

        # 결과 수정 버튼
        self.modify_btn = QPushButton("변경버튼")
        self.modify_btn.clicked.connect(self.modify)

        # 분리수거 설명 박스
        self.explain_box = QLabel("설명 예시입니다.")

        # box grid 생성
        # hbox 가로 / vbox 세로
        self.hbox1 = QHBoxLayout()
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.result_text)
        self.hbox1.addStretch(1)
        self.hbox1.addWidget(self.modify_btn)
        self.hbox1.addStretch(1)

        self.vbox1 = QVBoxLayout()
        self.vbox1.addStretch(1)
        self.vbox1.addWidget(self.image_box)
        self.vbox1.addWidget(self.img_upload_btn)
        self.vbox1.addLayout(self.hbox1)
        self.vbox1.addStretch(1)

        self.vbox3 = QVBoxLayout()
        self.vbox3.addWidget(self.explain_box)

        self.vbox2 = QVBoxLayout()

        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addLayout(self.vbox1)
        self.hbox.addStretch(1)
        self.hbox.addLayout(self.vbox3)
        self.hbox.addStretch(1)

        self.widget = QWidget()
        self.widget.setLayout(self.hbox)
        self.setCentralWidget(self.widget)


        self.setWindowTitle("recycle_project")
        # 모니터를 x, y축으로 구분, (모니터의 왼쪽 상단은 (0, 0))
        # x, y, 창의 넓이, 창의 높이 (좌표의 기준은 창의 왼쪽 상단)
        self.setGeometry(300, 100, 1500, 900)
        self.show()

    def img_upload(self):
        pass

    def modify(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())