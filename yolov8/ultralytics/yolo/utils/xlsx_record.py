from openpyxl import Workbook


def create_loss_xlsx():
    work_book = Workbook()  # 创建一个新的工作簿（workbook）
    work_sheet = work_book.active   # 选择默认的工作表（worksheet）
    # 数据可以直接分配到单元格中
    work_sheet['A1'] = "hard_loss"
    work_sheet['B1'] = "soft_loss"
    work_sheet['C1'] = "attention_loss"
    work_sheet['D1'] = "MGD_loss"
    work_sheet['E1'] = "CWD_loss"
    work_sheet['F1'] = "theta"
    work_sheet['G1'] = "beta"
    work_sheet['H1'] = "gamma"
    work_sheet['I1'] = "sigma"
    work_sheet['J1'] = "delta"
    return work_book, work_sheet


def append_date(hard_loss,
                soft_loss,
                attention_loss,
                mgd_loss,
                cwd_loss,
                theta,
                beta,
                gamma,
                sigma,
                delta,
                work_book,
                work_sheet,
                save_path):
    work_sheet.append([hard_loss, soft_loss, attention_loss, mgd_loss, cwd_loss, theta, beta, gamma, sigma, delta])
    try:
        work_book.save(save_path)
    except PermissionError:  # 避免xlsx文件打开就被终止
        return
