# visualizer.py
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np


class Visualizer:
    def __init__(self, master):
        """初始化可视化器"""
        self.master = master
        self.fig = None
        self.canvas = None
        self.current_data = None

    def create_visualization_frame(self):
        """创建可视化框架"""
        viz_frame = ttk.Frame(self.master)

        # 创建控制区域
        control_frame = ttk.LabelFrame(viz_frame, text="可视化控制", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 创建图表类型选择
        ttk.Label(control_frame, text="选择图表类型:").pack(side=tk.LEFT, padx=5)
        self.chart_type = ttk.Combobox(
            control_frame,
            values=['散点图对比', '残差图对比', '特征重要性对比', '箱线图',
                    '模型性能对比', '模型评估对比'],
            state='readonly'
        )
        self.chart_type.pack(side=tk.LEFT, padx=5)
        self.chart_type.set('散点图对比')
        self.chart_type.bind('<<ComboboxSelected>>', lambda e: self.refresh_plot())

        # 刷新按钮
        ttk.Button(control_frame, text="刷新图表",
                   command=self.refresh_plot).pack(side=tk.RIGHT, padx=5)

        # 创建图表显示区域
        plot_frame = ttk.Frame(viz_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 初始化matplotlib图形和画布
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 创建工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.X)

        return viz_frame

    def update_data(self, data_dict):
        """更新可视化数据"""
        self.current_data = data_dict
        self.refresh_plot()

    def refresh_plot(self):
        """刷新当前图表"""
        if not hasattr(self, 'current_data') or self.current_data is None:
            return

        chart_type = self.chart_type.get()
        try:
            if chart_type == '散点图对比':
                self.plot_scatter_comparison(
                    self.current_data['model_predictions'],
                    self.current_data['true_values'])
            elif chart_type == '残差图对比':
                self.plot_residuals_comparison(
                    self.current_data['model_predictions'],
                    self.current_data['true_values'])
            elif chart_type == '特征重要性对比':
                self.plot_feature_importance_comparison(
                    self.current_data['all_feature_importance'],
                    self.current_data['feature_names'])
            elif chart_type == '箱线图':
                self.plot_box(self.current_data['model_predictions'])
            elif chart_type == '模型性能对比':
                if 'model_metrics' in self.current_data:
                    self.plot_model_comparison(self.current_data['model_metrics'])
                else:
                    messagebox.showerror("错误", "无法获取模型性能指标数据")
            elif chart_type == '模型评估对比':
                if 'model_predictions' in self.current_data and 'true_values' in self.current_data:
                    self.plot_model_metrics_comparison(
                        self.current_data['model_predictions'],
                        self.current_data['true_values'])
                else:
                    messagebox.showerror("错误", "无法获取模型预测数据")
        except Exception as e:
            messagebox.showerror("错误", f"绘制图表时发生错误：\n{str(e)}")
            print(f"绘图错误: {str(e)}")

    def plot_scatter_comparison(self, model_predictions, true_values):
        """绘制多模型散点图对比"""
        self.set_plot_style()
        self.fig.clear()

        models = list(model_predictions.keys())
        n_models = len(models)
        n_cols = 2
        n_rows = (n_models + 1) // 2

        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
            ax.scatter(true_values, predictions, alpha=0.5)

            min_val = min(min(true_values), min(predictions))
            max_val = max(max(true_values), max(predictions))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测线')

            r_squared = np.corrcoef(true_values, predictions)[0, 1] ** 2
            ax.text(0.05, 0.95, f'R² = {r_squared:.4f}',
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title(f'{model_name}模型预测效果')
            ax.set_xlabel('实际值')
            ax.set_ylabel('预测值')
            ax.grid(True, linestyle='--', alpha=0.7)

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_residuals_comparison(self, model_predictions, true_values):
        """绘制多模型残差图对比"""
        self.set_plot_style()
        self.fig.clear()

        models = list(model_predictions.keys())
        n_models = len(models)
        n_cols = 2
        n_rows = (n_models + 1) // 2

        for i, (model_name, predictions) in enumerate(model_predictions.items()):
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
            residuals = np.array(predictions) - np.array(true_values)

            ax.scatter(predictions, residuals, alpha=0.5)
            ax.axhline(y=0, color='r', linestyle='--', label='零残差线')

            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            stats_text = f'均值: {mean_residual:.2f}\n标准差: {std_residual:.2f}'
            ax.text(0.05, 0.95, stats_text,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title(f'{model_name}模型残差分布')
            ax.set_xlabel('预测值')
            ax.set_ylabel('残差')
            ax.grid(True, linestyle='--', alpha=0.7)

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_model_comparison(self, model_metrics):
        """绘制模型性能对比图"""
        self.set_plot_style()
        self.fig.clear()

        # 创建子图
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_bar = self.fig.add_subplot(gs[0])
        ax_table = self.fig.add_subplot(gs[1])

        # 准备数据
        models = list(model_metrics.keys())
        metrics = ['MSE', 'RMSE', 'MAE']
        x = np.arange(len(models))
        width = 0.25

        # 绘制柱状图
        colors = ['#2196F3', '#4CAF50', '#FFC107']  # 蓝、绿、黄
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            values = [model_metrics[model][metric] for model in models]
            ax_bar.bar(x + width * (i - 1), values, width, label=metric, color=color)

        # 设置柱状图属性
        ax_bar.set_ylabel('误差值')
        ax_bar.set_title('模型性能指标对比')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(models)
        ax_bar.legend()

        # 添加网格线
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax_bar.set_axisbelow(True)

        # 准备表格数据
        table_data = []
        for model in models:
            row = [f"{model_metrics[model][metric]:.5f}" for metric in metrics]
            table_data.append(row)

        # 创建表格
        table = ax_table.table(cellText=table_data,
                               rowLabels=models,
                               colLabels=metrics,
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # 隐藏表格子图的坐标轴
        ax_table.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_box(self, data, labels=None):
        """绘制箱线图"""
        self.set_plot_style()
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        box_data = []
        model_names = []
        for model_name, predictions in data.items():
            box_data.append(predictions)
            model_names.append(model_name)

        bp = ax.boxplot(box_data, patch_artist=True, labels=model_names)

        colors = ['#2196F3', '#4CAF50', '#FFC107']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.setp(bp['whiskers'], color='black')
        plt.setp(bp['caps'], color='black')
        plt.setp(bp['medians'], color='red')
        plt.setp(bp['fliers'], marker='o', markerfacecolor='red', alpha=0.5)

        ax.set_ylabel('预测值分布')
        ax.set_title('各模型预测值分布箱线图')

        stats_text = "统计信息:\n"
        for i, predictions in enumerate(box_data):
            predictions_array = np.array(predictions)
            stats_text += f"{model_names[i]}:\n"
            stats_text += f"  均值: {np.mean(predictions_array):.2f}\n"
            stats_text += f"  中位数: {np.median(predictions_array):.2f}\n"
            stats_text += f"  标准差: {np.std(predictions_array):.2f}\n"

        ax.text(1.15, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.yaxis.grid(True, linestyle='--', alpha=0.7)

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_feature_importance_comparison(self, feature_importance_dict, feature_names):
        """绘制多模型特征重要性对比图"""
        self.set_plot_style()
        self.fig.clear()

        # 创建主绘图区和表格区
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
        ax_bar = self.fig.add_subplot(gs[0])
        ax_table = self.fig.add_subplot(gs[1])

        # 准备数据
        models = list(feature_importance_dict.keys())
        n_features = len(feature_names)
        x = np.arange(n_features)
        width = 0.25  # 柱状图宽度

        # 为每个模型绘制特征重要性条形图
        colors = ['#2196F3', '#4CAF50', '#FFC107']  # 蓝、绿、黄
        for i, (model_name, importance) in enumerate(zip(models, [feature_importance_dict[model] for model in models])):
            bars = ax_bar.bar(x + i * width - width, importance, width,
                              label=model_name, color=colors[i % len(colors)])

        # 设置图表属性
        ax_bar.set_ylabel('特征重要性')
        ax_bar.set_title('不同模型特征重要性对比')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(feature_names, rotation=45, ha='right')
        ax_bar.legend()
        ax_bar.grid(True, linestyle='--', alpha=0.7)

        # 准备表格数据
        table_data = []
        for model_name in models:
            importance = feature_importance_dict[model_name]
            # 获取前三个最重要的特征
            top_indices = np.argsort(importance)[-3:][::-1]
            top_features = [f"{feature_names[i]} ({importance[i]:.4f})" for i in top_indices]
            table_data.append([model_name] + top_features)

        # 创建表格
        column_labels = ['模型', '最重要特征 1', '最重要特征 2', '最重要特征 3']
        table = ax_table.table(cellText=table_data,
                               colLabels=column_labels,
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])

        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # 设置单元格颜色和样式
        for k, cell in table.get_celld().items():
            if k[0] == 0:  # 表头行
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#F0F0F0')
            cell.set_edgecolor('black')


        ax_table.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()

    def plot_model_metrics_comparison(self, model_predictions, true_values):
        """绘制模型评估指标（准确率、精确率、召回率）的对比图"""
        self.set_plot_style()
        self.fig.clear()

        # 创建两个子图：左边为柱状图，右边为表格
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
        ax_bar = self.fig.add_subplot(gs[0])
        ax_table = self.fig.add_subplot(gs[1])

        try:
            # 计算每个模型的指标
            metrics_data = {}
            threshold = np.median(true_values)  # 使用中位数作为分类阈值
            y_true_binary = (np.array(true_values) >= threshold).astype(int)

            for model_name, predictions in model_predictions.items():
                y_pred_binary = (np.array(predictions) >= threshold).astype(int)

                # 计算指标
                tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
                tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
                fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
                fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

                # 计算准确率、精确率和召回率
                accuracy = (tp + tn) / len(y_true_binary)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                metrics_data[model_name] = {
                    '准确率': accuracy,
                    '精确率': precision,
                    '召回率': recall
                }


            models = list(metrics_data.keys())
            metrics = ['准确率', '精确率', '召回率']
            x = np.arange(len(models))
            width = 0.25


            colors = ['#2196F3', '#4CAF50', '#FFC107']  # 蓝、绿、黄

            # 绘制柱状图
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                values = [metrics_data[model][metric] for model in models]
                ax_bar.bar(x + width * (i - 1), values, width, label=metric, color=color)


            ax_bar.set_ylabel('指标值')
            ax_bar.set_title('模型评估指标对比')
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(models)
            ax_bar.legend()
            ax_bar.grid(True, linestyle='--', alpha=0.7)

            
            table_data = []
            for model in models:
                row = [f"{metrics_data[model][metric]:.4f}" for metric in metrics]
                table_data.append(row)

            # 创建并设置表格
            table = ax_table.table(cellText=table_data,
                                   rowLabels=models,
                                   colLabels=metrics,
                                   loc='center',
                                   cellLoc='center')

            # 设置表格样式
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # 隐藏表格子图的坐标轴
            ax_table.axis('off')

            # 添加统计信息
            stats_text = "模型评估统计:\n"
            for model in models:
                stats_text += f"\n{model}:\n"
                for metric in metrics:
                    stats_text += f"  {metric}: {metrics_data[model][metric]:.4f}\n"

            # 在表格右侧添加统计信息
            ax_table.text(1.5, 0.5, stats_text,
                          transform=ax_table.transAxes,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                          verticalalignment='center')

            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            print(f"Metrics comparison plotting error: {str(e)}")
            messagebox.showerror("错误", f"绘制指标对比图时发生错误：\n{str(e)}")
            ax_bar.text(0.5, 0.5,
                        "无法绘制指标对比图\n请检查数据格式",
                        ha='center',
                        va='center',
                        transform=ax_bar.transAxes,
                        bbox=dict(boxstyle='round',
                                  facecolor='red',
                                  alpha=0.1))
            self.fig.tight_layout()
            self.canvas.draw()

    def set_plot_style(self):
        """设置全局图表风格"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans', 'Arial'],
            'axes.unicode_minus': False,
            'figure.autolayout': True,
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.7,
            'axes.labelsize': 10,
            'axes.titlesize': 12,
            'font.size': 10,
            'lines.linewidth': 2,
            'lines.markersize': 6,
            'legend.fontsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.titlesize': 14,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.linewidth': 1.2,
            'grid.color': '#cccccc'
        })

    def clear_plot(self):
        """清除当前图表"""
        if self.fig is not None:
            self.fig.clear()
            self.canvas.draw()

    def save_plot(self, filename):
        """保存当前图表"""
        try:
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            messagebox.showerror("错误", f"保存图表时发生错误：\n{str(e)}")
            return False