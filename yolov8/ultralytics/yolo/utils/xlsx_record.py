from openpyxl import Workbook


def create_loss_xlsx():
    work_book = Workbook()  # 创建一个新的工作簿（workbook）
    work_sheet = work_book.active   # 选择默认的工作表（worksheet）
    # 数据可以直接分配到单元格中
    work_sheet['A1'] = "hard_loss"
    work_sheet['B1'] = "soft_loss"
    work_sheet['C1'] = "attention_loss"
    work_sheet['D1'] = "theta"
    work_sheet['E1'] = "beta"
    work_sheet['F1'] = "gamma"
    return work_book, work_sheet


def append_date(hard_loss,
                soft_loss,
                attention_loss,
                theta,
                beta,
                gamma,
                work_book,
                work_sheet,
                save_path):
    work_sheet.append([hard_loss, soft_loss, attention_loss, theta, beta, gamma])
    work_book.save(save_path)

