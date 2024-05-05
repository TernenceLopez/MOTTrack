import pandas as pd
import matplotlib.pyplot as plt


def plot_excel_column(file_path, column_num):
    """
    读取 Excel 文件中指定列的数据，并绘制成曲线图。

    参数：
    file_path (str): Excel 文件的路径。
    column_name (str): 要绘制的列。

    返回：
    None
    """
    # 读取 Excel 文件
    data_file = pd.read_excel(file_path)
    # 获取指定列的数据
    data = data_file[data_file.columns[column_num]]
    # 绘制曲线图
    plt.plot(data)
    plt.xlabel('Index')
    plt.ylabel(data_file.columns[column_num])
    plt.title(f'{data_file.columns[column_num]}')
    plt.show()


def plot_excel_total(file_path):
    data_file = pd.read_excel(file_path)
    # 获取列数
    num_cols = data_file.columns.size
    # 创建 subplot 布局
    num_rows = num_cols // 2 + num_cols % 2  # 根据列数计算行数
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6 * num_rows))
    # 扁平化子图数组
    axs = axs.flatten()
    # 遍历每一列，绘制成子图
    for i, col_name in enumerate(data_file.columns):
        axs[i].plot(data_file[col_name])
        axs[i].set_title(col_name)
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel(col_name)
    # 调整子图布局
    plt.tight_layout()
    plt.show()


def plot_csv_all_columns(file_path):
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    # 获取列数和行数
    num_cols = len(df.columns)
    # 创建 subplot 布局
    num_rows = num_cols // 2 + num_cols % 2  # 根据列数计算行数
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
    # 扁平化子图数组
    axs = axs.flatten()
    # 遍历每一列，绘制成子图
    for i, col_name in enumerate(df.columns):
        axs[i].plot(df[col_name])
        axs[i].set_title(col_name)
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel(col_name)
    # 调整子图布局
    plt.tight_layout()
    plt.show()


def plot_selected_columns_by_index(file_path, column_indices):
    """
    读取 CSV 文件中指定索引的列的数据，并绘制成子图。
    参数：
    file_path (str): CSV 文件的路径。
    column_indices (list): 要绘制的列的索引列表。
    返回：
    None
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    # 获取所有列的名称
    column_names = df.columns.tolist()
    # 创建 subplot 布局
    num_plots = len(column_indices)
    num_rows = num_plots // 2 + num_plots % 2  # 根据列数计算行数
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 6*num_rows))
    # 扁平化子图数组
    axs = axs.flatten()
    # 遍历每一列索引，绘制成子图
    for i, col_idx in enumerate(column_indices):
        col_name = column_names[col_idx]
        axs[i].plot(df.iloc[:, col_idx])
        axs[i].set_title(col_name)
        axs[i].set_xlabel('Index')
        axs[i].set_ylabel(col_name)
    # 调整子图布局
    plt.tight_layout()
    plt.show()

