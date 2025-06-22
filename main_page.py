import sys
from PyQt5.QtWidgets import QMainWindow, QApplication

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("recycle_project")
        # 모니터를 x, y축으로 구분, (모니터의 왼쪽 상단은 (0, 0))
        # x, y, 창의 넓이, 창의 높이 (좌표의 기준은 창의 왼쪽 상단)
        self.setGeometry(300, 100, 1500, 900)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())