#main_system.py
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# 导入自定义模块
from login import LoginWindow
from data_processor import DataProcessor
from models import LinearRegression, DecisionTreeRegressor, KNNRegressor
from visualizer import Visualizer


class MainSystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("学生成绩预测系统")
        self.root.geometry("1200x800")

        # 初始时隐藏窗口
        self.root.withdraw()

        # 初始化数据和模型
        self.data_processor = DataProcessor()
        self.data = None
        self.processed_data = None
        self.model = None
        self.predictions = None

        # 创建主界面
        self.create_main_interface()

        # 窗口居中显示
        self.center_window()

        # 配置主窗口网格权重
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

    def show(self):
        """显示主窗口"""
        # 显示主窗口
        self.root.deiconify()
        # 重新调整窗口位置
        self.center_window()
        # 将窗口提到前台
        self.root.lift()
        # 确保窗口获得焦点
        self.root.focus_force()

    def center_window(self):
        """窗口居中显示"""
        # 等待窗口更新几何信息
        self.root.update_idletasks()

        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # 获取窗口尺寸
        window_width = 1200
        window_height = 800

        # 计算居中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # 设置窗口位置
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def create_main_interface(self):
        """创建主界面"""
        # 创建菜单栏
        self.create_menu()

        # 创建标题栏
        title_frame = ttk.Frame(self.root)
        title_frame.grid(row=0, column=0, sticky='ew', padx=10, pady=5)
        ttk.Label(title_frame, text="学生成绩预测系统",
                  font=('Helvetica', 16, 'bold')).pack()

        # 创建notebook用于切换不同功能页面
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)

        # 创建各功能页面
        self.data_import_frame = self.create_data_import_page()
        self.preprocessing_frame = self.create_preprocessing_page()
        self.model_frame = self.create_model_training_page()
        self.visualization_frame = self.create_visualization_page()


        self.notebook.add(self.data_import_frame, text='数据导入')
        self.notebook.add(self.preprocessing_frame, text='数据预处理')
        self.notebook.add(self.model_frame, text='模型训练')
        self.notebook.add(self.visualization_frame, text='结果可视化')


        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, sticky='ew')

    def create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="导入数据", command=self.import_data)
        file_menu.add_command(label="保存模型", command=self.save_model)
        file_menu.add_command(label="加载模型", command=self.load_model)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="使用说明", command=self.show_help)
        help_menu.add_command(label="关于", command=self.show_about)

    def create_data_import_page(self):
        """创建数据导入页面"""
        frame = ttk.Frame(self.notebook, padding="10")

        # 文件选择区域
        file_frame = ttk.LabelFrame(frame, text="数据文件", padding="5")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="选择文件",
                   command=self.import_data).pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(file_frame, text="未选择文件")
        self.file_label.pack(side=tk.LEFT, padx=5)


        preview_frame = ttk.LabelFrame(frame, text="数据预览", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        columns = ('Column', 'Type', 'Sample')
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show='headings')


        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100)


        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL,
                                  command=self.preview_tree.yview)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)


        self.preview_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        return frame

    def create_preprocessing_page(self):
        """创建数据预处理页面"""
        frame = ttk.Frame(self.notebook, padding="10")

        # 预处理选项
        options_frame = ttk.LabelFrame(frame, text="预处理选项", padding="5")
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # 目标变量选择
        target_frame = ttk.Frame(options_frame)
        target_frame.pack(fill=tk.X, pady=5)
        ttk.Label(target_frame, text="目标变量:").pack(side=tk.LEFT, padx=5)
        self.target_var = tk.StringVar(value="G3")  # 设置默认值
        self.target_combo = ttk.Combobox(target_frame, textvariable=self.target_var, state='readonly')
        self.target_combo['values'] = ['G3']  # 设置初始可选值
        self.target_combo.set('G3')  # 设置默认选中值
        self.target_combo.pack(side=tk.LEFT, padx=5)

        # 预处理选项
        self.standardize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="特征标准化",
                        variable=self.standardize_var).pack(anchor=tk.W, pady=2)

        self.missing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="处理缺失值",
                        variable=self.missing_var).pack(anchor=tk.W, pady=2)

        # 执行按钮
        ttk.Button(options_frame, text="执行预处理",
                   command=self.run_preprocessing).pack(pady=10)


        result_frame = ttk.LabelFrame(frame, text="预处理结果", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        columns = ('Feature', 'Original', 'Processed')
        self.preprocess_tree = ttk.Treeview(result_frame, columns=columns, show='headings')

        for col in columns:
            self.preprocess_tree.heading(col, text=col)
            self.preprocess_tree.column(col, width=100)


        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                  command=self.preprocess_tree.yview)
        self.preprocess_tree.configure(yscrollcommand=scrollbar.set)

        self.preprocess_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        return frame

    def create_model_training_page(self):
        """创建模型训练页面"""
        frame = ttk.Frame(self.notebook, padding="10")

        # 模型配置区域
        config_frame = ttk.LabelFrame(frame, text="模型配置", padding="5")
        config_frame.pack(fill=tk.X, padx=5, pady=5)

        # 模型选择
        model_frame = ttk.Frame(config_frame)
        model_frame.pack(fill=tk.X, pady=5)

        ttk.Label(model_frame, text="选择模型:").pack(side=tk.LEFT, padx=5)
        self.model_var = tk.StringVar(value="线性回归")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                   values=['线性回归', '决策树', 'KNN'], state='readonly')
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_select)

        # 模型参数配置
        self.param_frames = {}

        # 线性回归参数
        lr_frame = ttk.Frame(config_frame)
        self.param_frames['线性回归'] = lr_frame
        ttk.Label(lr_frame, text="学习率:").grid(row=0, column=0, padx=5, pady=2)
        self.lr_learning_rate = ttk.Entry(lr_frame, width=10)
        self.lr_learning_rate.insert(0, "0.01")
        self.lr_learning_rate.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(lr_frame, text="迭代次数:").grid(row=0, column=2, padx=5, pady=2)
        self.lr_iterations = ttk.Entry(lr_frame, width=10)
        self.lr_iterations.insert(0, "1000")
        self.lr_iterations.grid(row=0, column=3, padx=5, pady=2)

        # 决策树参数
        dt_frame = ttk.Frame(config_frame)
        self.param_frames['决策树'] = dt_frame
        ttk.Label(dt_frame, text="最大深度:").grid(row=0, column=0, padx=5, pady=2)
        self.dt_max_depth = ttk.Entry(dt_frame, width=10)
        self.dt_max_depth.insert(0, "5")
        self.dt_max_depth.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(dt_frame, text="最小分裂样本数:").grid(row=0, column=2, padx=5, pady=2)
        self.dt_min_samples = ttk.Entry(dt_frame, width=10)
        self.dt_min_samples.insert(0, "2")
        self.dt_min_samples.grid(row=0, column=3, padx=5, pady=2)

        # KNN参数
        knn_frame = ttk.Frame(config_frame)
        self.param_frames['KNN'] = knn_frame
        ttk.Label(knn_frame, text="K值:").grid(row=0, column=0, padx=5, pady=2)
        self.knn_k = ttk.Entry(knn_frame, width=10)
        self.knn_k.insert(0, "3")
        self.knn_k.grid(row=0, column=1, padx=5, pady=2)

        # 默认显示线性回归参数
        self.param_frames['线性回归'].pack(fill=tk.X, pady=5)

        # 训练按钮
        ttk.Button(config_frame, text="训练模型",
                   command=self.train_model).pack(pady=10)

        # 训练结果显示
        result_frame = ttk.LabelFrame(frame, text="训练结果", padding="5")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)


        self.train_result_text = tk.Text(result_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                  command=self.train_result_text.yview)
        self.train_result_text.configure(yscrollcommand=scrollbar.set)

        self.train_result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        return frame

    def create_visualization_page(self):
        """创建可视化页面"""
        # 创建一个frame作为可视化页面的容器
        frame = ttk.Frame(self.notebook, padding="10")

        self.visualizer = Visualizer(frame)


        viz_frame = self.visualizer.create_visualization_frame()
        viz_frame.pack(fill=tk.BOTH, expand=True)

        return frame

    def on_model_select(self, event=None):
        """模型选择改变时的处理"""
        selected_model = self.model_var.get()

        # 隐藏所有参数框架
        for frame in self.param_frames.values():
            frame.pack_forget()

        # 显示选中模型的参数框架
        self.param_frames[selected_model].pack(fill=tk.X, pady=5)

    def import_data(self):
        """导入数据"""
        try:
            filename = filedialog.askopenfilename(
                title='选择数据文件',
                filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
            )

            if filename:
                # 更新文件标签
                self.file_label.config(text=os.path.basename(filename))

                # 加载数据
                self.data, headers = self.data_processor.load_data(filename)

                # 更新目标变量选择
                self.target_combo['values'] = headers
                if 'G3' in headers:
                    self.target_combo.set('G3')

                # 更新数据预览
                self.update_data_preview()

                self.status_var.set("数据导入成功")
                messagebox.showinfo("成功", "数据导入成功！")
        except Exception as e:
            self.status_var.set("数据导入失败")
            messagebox.showerror("错误", f"导入数据时发生错误：\n{str(e)}")

    def update_data_preview(self):
        """更新数据预览表格"""
        # 清空现有内容
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)

        if self.data:
            for col in self.data:
                # 确定列类型
                try:
                    float(self.data[col][0])
                    col_type = "Numeric"
                except:
                    col_type = "Categorical"

                # 获取样本值
                sample = str(self.data[col][0])
                if len(sample) > 20:
                    sample = sample[:20] + "..."

                # 添加到表格
                self.preview_tree.insert('', 'end', values=(col, col_type, sample))

    def run_preprocessing(self):
        """执行数据预处理"""
        if not self.data:
            messagebox.showerror("错误", "请先导入数据！")
            return

        try:
            # 检查目标变量是否存在于数据中
            target = self.target_var.get()
            if target not in self.data:
                messagebox.showerror("错误", f"数据中不存在目标变量 {target}！")
                return

            # 执行预处理
            self.processed_data = self.data_processor.preprocess_data(
                self.data,
                standardize=self.standardize_var.get(),
                handle_missing=self.missing_var.get()
            )

            self.update_preprocessing_result()
            self.status_var.set("数据预处理完成")
            messagebox.showinfo("成功", "数据预处理完成！")
        except Exception as e:
            self.status_var.set("数据预处理失败")
            messagebox.showerror("错误", f"预处理数据时发生错误：\n{str(e)}")

    def update_preprocessing_result(self):
        """更新预处理结果显示"""
        # 清空现有内容
        for item in self.preprocess_tree.get_children():
            self.preprocess_tree.delete(item)

        if self.data and self.processed_data:
            for col in self.data:
                # 获取原始值和处理后的值
                orig_val = str(self.data[col][0])
                proc_val = str(self.processed_data[col][0])

                # 截断过长的值
                if len(orig_val) > 20:
                    orig_val = orig_val[:20] + "..."
                if len(proc_val) > 20:
                    proc_val = proc_val[:20] + "..."

                # 添加到表格
                self.preprocess_tree.insert('', 'end', values=(col, orig_val, proc_val))

    def train_model(self):
        """训练并评估模型"""
        if not self.processed_data:
            messagebox.showerror("错误", "请先完成数据预处理！")
            return

        try:
            # 准备训练数据
            features, target = self.data_processor.split_features_target(self.processed_data)
            X = self.data_processor.convert_dict_to_matrix(features)
            y = [float(val) for val in target]
            feature_names = self.data_processor.get_feature_names(self.processed_data)

            # 存储所有模型的评估结果
            model_metrics = {}
            model_predictions = {}
            feature_importance = {}

            # 训练和评估线性回归模型
            lr_model = LinearRegression(
                learning_rate=float(self.lr_learning_rate.get()),
                n_iterations=int(self.lr_iterations.get())
            )
            lr_model.fit(X, y)
            lr_metrics, lr_predictions = lr_model.evaluate(X, y)
            model_metrics['线性回归'] = lr_metrics
            model_predictions['线性回归'] = lr_predictions
            feature_importance['线性回归'] = lr_model.feature_importances_

            # 训练和评估决策树模型
            dt_model = DecisionTreeRegressor(
                max_depth=int(self.dt_max_depth.get()),
                min_samples_split=int(self.dt_min_samples.get())
            )
            dt_model.fit(X, y)
            dt_metrics, dt_predictions = dt_model.evaluate(X, y)
            model_metrics['决策树'] = dt_metrics
            model_predictions['决策树'] = dt_predictions
            feature_importance['决策树'] = dt_model.feature_importances_

            # 训练和评估KNN模型
            knn_model = KNNRegressor(
                k=int(self.knn_k.get())
            )
            knn_model.fit(X, y)
            knn_metrics, knn_predictions = knn_model.evaluate(X, y)
            model_metrics['KNN'] = knn_metrics
            model_predictions['KNN'] = knn_predictions
            feature_importance['KNN'] = knn_model.feature_importances_

            # 更新训练结果显示
            self.update_training_results(model_metrics)

            # 准备可视化数据
            viz_data = {
                'true_values': y,
                'current_predictions': lr_predictions,  # 默认显示线性回归的预测结果
                'model_predictions': model_predictions,
                'model_metrics': model_metrics,
                'feature_importance': lr_model.feature_importances_,  # 默认显示线性回归的特征重要性
                'feature_names': feature_names,
                'all_feature_importance': feature_importance
            }

            # 更新可视化
            self.visualizer.update_data(viz_data)

            # 保存当前模型状态
            self.current_model_state = {
                'linear_regression': lr_model,
                'decision_tree': dt_model,
                'knn': knn_model
            }

            self.status_var.set("模型训练完成")
            messagebox.showinfo("成功", "模型训练完成！")

        except Exception as e:
            self.status_var.set("模型训练失败")
            messagebox.showerror("错误", f"训练模型时发生错误：\n{str(e)}")

    def update_training_results(self, model_metrics):
        """更新训练结果显示"""
        result_text = "模型训练结果比较：\n\n"

        # 为每个模型添加详细的评估指标
        for model_name, metrics in model_metrics.items():
            result_text += f"{model_name}模型评估指标：\n"
            result_text += f"  MSE: {metrics['MSE']:.4f}\n"
            result_text += f"  RMSE: {metrics['RMSE']:.4f}\n"
            result_text += f"  MAE: {metrics['MAE']:.4f}\n"
            result_text += f"  R²: {metrics['R2']:.4f}\n\n"

        # 找出最佳模型
        best_model = min(model_metrics.items(), key=lambda x: x[1]['MSE'])
        result_text += f"\n最佳模型: {best_model[0]}\n"
        result_text += f"最佳MSE: {best_model[1]['MSE']:.4f}\n"

        # 更新文本显示
        self.train_result_text.delete('1.0', tk.END)
        self.train_result_text.insert('1.0', result_text)

    def save_model(self):
        """保存当前模型"""
        if not hasattr(self, 'current_model_state'):
            messagebox.showerror("错误", "没有可保存的模型！")
            return

        try:
            filename = filedialog.asksaveasfilename(
                title='保存模型',
                defaultextension='.json',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
            )

            if filename:
                model_data = {
                    'linear_regression': {
                        'weights': self.current_model_state['linear_regression'].weights,
                        'bias': self.current_model_state['linear_regression'].bias
                    },
                    'decision_tree': {
                        'max_depth': self.current_model_state['decision_tree'].max_depth,
                        'min_samples_split': self.current_model_state['decision_tree'].min_samples_split
                    },
                    'knn': {
                        'k': self.current_model_state['knn'].k
                    }
                }

                with open(filename, 'w') as f:
                    json.dump(model_data, f)

                self.status_var.set("模型保存成功")
                messagebox.showinfo("成功", "模型保存成功！")

        except Exception as e:
            self.status_var.set("模型保存失败")
            messagebox.showerror("错误", f"保存模型时发生错误：\n{str(e)}")

    def load_model(self):
        """加载保存的模型"""
        try:
            filename = filedialog.askopenfilename(
                title='加载模型',
                filetypes=[('JSON files', '*.json'), ('All files', '*.*')]
            )

            if filename:
                with open(filename, 'r') as f:
                    model_data = json.load(f)

                # 重建线性回归模型
                lr_model = LinearRegression()
                lr_model.weights = model_data['linear_regression']['weights']
                lr_model.bias = model_data['linear_regression']['bias']

                # 重建决策树模型
                dt_model = DecisionTreeRegressor(
                    max_depth=model_data['decision_tree']['max_depth'],
                    min_samples_split=model_data['decision_tree']['min_samples_split']
                )

                # 重建KNN模型
                knn_model = KNNRegressor(
                    k=model_data['knn']['k']
                )

                # 更新当前模型状态
                self.current_model_state = {
                    'linear_regression': lr_model,
                    'decision_tree': dt_model,
                    'knn': knn_model
                }

                # 更新界面参数
                self.lr_learning_rate.delete(0, tk.END)
                self.lr_learning_rate.insert(0, "0.01")

                self.dt_max_depth.delete(0, tk.END)
                self.dt_max_depth.insert(0, str(dt_model.max_depth))

                self.dt_min_samples.delete(0, tk.END)
                self.dt_min_samples.insert(0, str(dt_model.min_samples_split))

                self.knn_k.delete(0, tk.END)
                self.knn_k.insert(0, str(knn_model.k))

                self.status_var.set("模型加载成功")
                messagebox.showinfo("成功", "模型加载成功！")

        except Exception as e:
            self.status_var.set("模型加载失败")
            messagebox.showerror("错误", f"加载模型时发生错误：\n{str(e)}")

    def create_visualization_frame(self):
        """创建可视化区域"""
        return self.visualizer.create_visualization_frame()

    def show_help(self):
        """显示帮助信息"""
        help_text = """
        学生成绩预测系统使用说明：

        1. 数据导入
           - 支持CSV格式的学生成绩数据
           - 数据预处理可自动处理缺失值和标准化

        2. 模型训练
           - 提供线性回归、决策树和KNN三种模型
           - 可调整各模型的超参数
           - 支持模型性能对比和评估

        3. 结果可视化
           - 散点图：显示预测值与实际值的对比
           - 残差图：分析预测误差分布
           - 特征重要性：展示各特征对预测的影响
           - ROC曲线：对比不同模型的性能
           - 支持图表交互和保存

        4. 模型保存/加载
           - 可保存训练好的模型参数
           - 支持加载已保存的模型继续使用
        """
        messagebox.showinfo("使用说明", help_text)

    def show_about(self):
        """显示关于信息"""
        about_text = """
        学生成绩预测系统 v2.0

        本系统实现了基于机器学习的学生成绩预测功能：

        主要特点：
        - 支持多种机器学习模型
        - 提供丰富的数据可视化功能
        - 包含完整的模型评估体系
        - 简洁直观的用户界面

        技术特性：
        - 使用Python开发
        - 基于Tkinter构建GUI
        - 支持数据预处理和特征工程
        - 实现了模型性能对比分析

        开发完成时间：2024年
        """
        messagebox.showinfo("关于", about_text)

    def run(self):
        """运行主系统"""
        self.root.mainloop()