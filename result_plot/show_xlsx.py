from utils.xlsx_plot import *

if __name__ == "__main__":
    # 调用方法，绘制指定 Excel 文件中指定列的数据曲线图
    # plot_excel_column('./xlsx_dir/loss_record.xlsx', 1)
    plot_excel_total('./xlsx_dir/loss_record.xlsx')

    # 调用方法，绘制指定 CSV 文件中指定列的数据曲线图
    # plot_csv_all_columns("./xlsx_dir/results.csv")
    plot_selected_columns_by_index("./xlsx_dir/results.csv",
                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
