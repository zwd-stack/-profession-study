# main.py
from login import LoginWindow
from main_system import MainSystem


def main():

    main_system = MainSystem()

    # 创建登录窗口
    login = LoginWindow()

    def start_main_system():
        """登录成功后的回调函数"""
        # 显示主系统窗口
        main_system.show()
        # 运行主系统
        main_system.run()

    # 设置登录成功后的回调
    login.set_main_system_callback(start_main_system)

    # 运行登录窗口
    login.run()


if __name__ == "__main__":
    main()
