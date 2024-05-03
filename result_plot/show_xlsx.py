from utils.xlsx_plot import *

if __name__ == "__main__":
    # 调用方法，绘制指定 Excel 文件中指定列的数据曲线图
    # plot_excel_column('./xlsx_dir/loss_record.xlsx', 1)
    plot_excel_total('./xlsx_dir/loss_record.xlsx')
