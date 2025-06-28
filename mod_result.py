import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QRadioButton, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import QCoreApplication

class Modify(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.recycle_type1 = QRadioButton("일반쓰레기")
        self.recycle_type2 = QRadioButton("페트")
        self.recycle_type3 = QRadioButton("플라스틱")
        self.recycle_type4 = QRadioButton("비닐")

        self.submitbtn = QPushButton("제출")
        self.submitbtn.clicked.connect(self.submit)
        self.cancelbtn = QPushButton("취소")
        self.cancelbtn.clicked.connect(QCoreApplication.instance().quit)
        
        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.submitbtn)
        self.hbox.addWidget(self.cancelbtn)

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.recycle_type1)
        self.vbox.addWidget(self.recycle_type2)
        self.vbox.addWidget(self.recycle_type3)
        self.vbox.addWidget(self.recycle_type4)
        self.vbox.addLayout(self.hbox)

        self.widget = QWidget()
        self.widget.setLayout(self.vbox)
        self.setCentralWidget(self.widget)

        self.setWindowTitle("결과 수정")
        self.setGeometry(400, 400, 400, 700)
        self.show()
    
    # 두 번 클릭해야 종료됨 다른 방법 필요
    def submit(self):
        self.submitbtn.clicked.connect(QCoreApplication.instance().quit)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Modify()
    sys.exit(app.exec_())