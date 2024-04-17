import os


def generate(dir, folder):
    files = os.listdir(dir)
    files.sort()
    print('****************')
    print('input :', dir)
    print('start...')
    listText = open(str(folder) + '.txt', 'a')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = './images/' + str(folder) + '/' + file + '\n'
        listText.write(name)
    listText.close()
    print('down!')
    print('****************')


if __name__ == '__main__':
    outer_path = './images'  # 这里是你的图片的目录
    i = 0
    folderlist = os.listdir(outer_path)  # 列举文件夹
    for folder in folderlist:
        generate(os.path.join(outer_path, folder), folder)
        i += 1
