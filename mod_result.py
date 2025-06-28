import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QRadioButton, QWidget, QVBoxLayout

class Modify(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.recycle_type1 = QRadioButton("일반쓰레기")
        self.recycle_type2 = QRadioButton("페트")
        self.recycle_type3 = QRadioButton("플라스틱")
        self.recycle_type4 = QRadioButton("비닐")

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.recycle_type1)
        self.vbox.addWidget(self.recycle_type2)
        self.vbox.addWidget(self.recycle_type3)
        self.vbox.addWidget(self.recycle_type4)

        self.widget = QWidget()
        self.widget.setLayout(self.vbox)
        self.setCentralWidget(self.widget)

        self.setWindowTitle("결과 수정")
        self.setGeometry(400, 400, 400, 700)
        self.show()

if __name__ == "mod_result":
    app = QApplication(sys.argv)
    ex = Modify()
    sys.exit(app.exec_())