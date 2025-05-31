#login.py
import tkinter as tk
from tkinter import ttk, messagebox
import json
import os


class LoginWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("学生成绩预测系统 - 登录")
        self.root.geometry("400x350")
        self.users_file = "users.json"
        self.load_users()
        self.setup_ui()
        # 保存主系统回调函数
        self.main_system_callback = None

    def load_users(self):
        """加载用户数据"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            self.users = {"admin": "123456"}
            self.save_users()

    def save_users(self):
        """保存用户数据"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f)

    def setup_ui(self):
        """设置登录界面UI"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = ttk.Label(main_frame, text="学生成绩预测系统", font=('Helvetica', 16, 'bold'))
        title_label.pack(pady=20)

        # 用户名框架
        username_frame = ttk.Frame(main_frame)
        username_frame.pack(fill=tk.X, pady=5)
        ttk.Label(username_frame, text="用户名：").pack(side=tk.LEFT)
        self.username = ttk.Entry(username_frame)
        self.username.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 密码框架
        password_frame = ttk.Frame(main_frame)
        password_frame.pack(fill=tk.X, pady=5)
        ttk.Label(password_frame, text="密码：").pack(side=tk.LEFT)
        self.password = ttk.Entry(password_frame, show="*")
        self.password.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        # 登录按钮
        login_button = ttk.Button(button_frame, text="登录", command=self.login)
        login_button.pack(side=tk.LEFT, padx=10)

        # 注册按钮
        register_button = ttk.Button(button_frame, text="注册", command=self.show_register_window)
        register_button.pack(side=tk.LEFT, padx=10)

        # 状态标签
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.pack()

        # 绑定回车键
        self.username.bind('<Return>', lambda e: self.password.focus())
        self.password.bind('<Return>', lambda e: self.login())

        # 设置初始焦点
        self.username.focus()

    def login(self):
        """登录验证"""
        username = self.username.get().strip()
        password = self.password.get().strip()

        if not username or not password:
            self.status_label.config(text="请输入用户名和密码", foreground="red")
            return

        if username in self.users and self.users[username] == password:
            self.status_label.config(text="登录成功", foreground="green")
            self.root.after(500, self.login_success)
        else:
            self.status_label.config(text="用户名或密码错误", foreground="red")
            self.password.delete(0, tk.END)

    def show_register_window(self):
        """显示注册窗口"""
        self.register_window = tk.Toplevel(self.root)
        self.register_window.title("注册新用户")
        self.register_window.geometry("400x350")
        self.register_window.transient(self.root)

        # 注册窗口界面
        register_frame = ttk.Frame(self.register_window, padding="20")
        register_frame.pack(fill=tk.BOTH, expand=True)

        # 用户名输入
        ttk.Label(register_frame, text="用户名：").pack(pady=5)
        self.reg_username = ttk.Entry(register_frame)
        self.reg_username.pack(fill=tk.X)

        # 密码输入
        ttk.Label(register_frame, text="密码：").pack(pady=5)
        self.reg_password = ttk.Entry(register_frame, show="*")
        self.reg_password.pack(fill=tk.X)


        ttk.Label(register_frame, text="确认密码：").pack(pady=5)
        self.reg_confirm = ttk.Entry(register_frame, show="*")
        self.reg_confirm.pack(fill=tk.X)


        button_frame = ttk.Frame(register_frame)
        button_frame.pack(pady=30)


        confirm_btn = ttk.Button(button_frame, text="确认", command=self.register, style='Large.TButton')
        confirm_btn.pack(side=tk.LEFT, padx=20)


        cancel_btn = ttk.Button(button_frame, text="取消", command=self.register_window.destroy, style='Large.TButton')
        cancel_btn.pack(side=tk.LEFT, padx=20)


        style = ttk.Style()
        style.configure('Large.TButton', padding=(20, 10))


        self.reg_status = ttk.Label(register_frame, text="")
        self.reg_status.pack()

    def register(self):
        """处理注册请求"""
        username = self.reg_username.get().strip()
        password = self.reg_password.get().strip()
        confirm = self.reg_confirm.get().strip()

        if not username or not password or not confirm:
            self.reg_status.config(text="所有字段都必须填写", foreground="red")
            return

        if password != confirm:
            self.reg_status.config(text="两次输入的密码不一致", foreground="red")
            return

        if username in self.users:
            self.reg_status.config(text="用户名已存在", foreground="red")
            return

        self.users[username] = password
        self.save_users()

        self.reg_status.config(text="注册成功！", foreground="green")
        self.root.after(1000, self.register_window.destroy)

    def login_success(self):
        """登录成功后的处理"""
        self.root.withdraw()
        if self.main_system_callback:
            self.main_system_callback()

    def set_main_system_callback(self, callback):
        """设置主系统启动回调"""
        self.main_system_callback = callback

    def run(self):
        self.root.mainloop()