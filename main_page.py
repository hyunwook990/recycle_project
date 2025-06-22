import sys
from PyQt5.QtWidgets import QMainWindow, QApplication

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recycle()
    
    def recycle(self):


        self.setWindowTitle("recycle_project")
        self.setGeometry(300, 300, 300, 300)
        self.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())