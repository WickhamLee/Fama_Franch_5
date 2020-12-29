# 目的： 遍历文件夹中的内容，并进行处理

import os
import shutil

# 返回文件夹中的所有文件列表，包含子文件夹
# 输出： file_path_list： 完整路径列表
#        file_name_list： 文件名列表
#        sub_path_list：  不包含母文件夹的文件路径列表，不包含最终文件名
def get_all_subfolder_files(folder_path):
    file_path_list = []
    file_name_list = []
    sub_path_list = []
    for subfolder_path, _, file_list_in_subfolder, in os.walk(folder_path):
        for file_name in file_list_in_subfolder:
            file_path_list.append(subfolder_path + '\\' + file_name)
            file_name_list.append(file_name)
            sub_path_list.append(subfolder_path.replace(folder_path, '') + '\\')
    return file_path_list, file_name_list, sub_path_list


# 返回文件夹中的所有文件列表，不b包含子文件夹
# 输出： file_path_list： 完整路径列表
#        file_name_list： 文件名列表


def get_all_folder_files(folder_path):
    # folder_path 尾部需要加上\\
    if folder_path[-1] != '\\':
        folder_path = folder_path + '\\'
    file_names_list = os.listdir(folder_path)
    file_path_list = []
    for file_name in file_names_list:
        file_path_list.append(folder_path + file_name)

    return file_path_list, file_names_list



#  从右边往左边搜索起，直到遇到第一个点为止。将右边的字符理解为后缀。
def get_file_extension(file_name):
    file_extension = ''
    file_name_proper = file_name
    file_len = len(file_name)
    for i in range(file_len):
        if file_name[-i] == '.':
            file_extension = file_name[- i + 1:]
            file_name_proper = file_name[0: file_len - i]
            break
    return file_extension, file_name_proper


# 从完整文件路径中分离中文件名
def get_file_name_from_path(file_path):
    file_name = ''
    for i in range(len(file_path)):
        if file_path[-i] == '\\':
            file_name = file_path[- i + 1:]
            break
    return file_name

# 从完整文件路径中分离中文件夹路径
def get_folder_name_from_path(file_path):
    folder_name = ''
    for i in range(len(file_path)):
        if file_path[-i] == '\\':
            folder_name = file_path[:len(file_path) - i]
            break
    return folder_name



# 删除文件夹内的内容
def delete_folder_content(folder_path, ignore_sub_folders = True):
    if ignore_sub_folders:
        file_path_list,_ = get_all_folder_files(folder_path)
        for file_path in file_path_list:
            if os.path.isfile(file_path):
                os.remove(file_path)
        return True
    else:
        raise Exception('delete_folder_content: 暂时没有实现删除所有的子文件夹内的文件的功能')


# ---------------------------------------------------
# 检查文件夹是否存在，如果不存在的话，创建一个文件夹
# ---------------------------------------------------

def create_folder(folder_path):

    if not os.path.exists(folder_path):

        folder_list = folder_path.split('\\')
        current_path = ''
        for i in range(len(folder_list)):
            if i == 0:
                current_path = folder_list[0]
            else:
                current_path = current_path + '\\' + folder_list[i]

            # Python需要一层一层创建
            if not os.path.exists(current_path):                            # 这里只需要检查一次，不过这个地方无所谓
                os.mkdir(current_path)

        output = 'New folder created at '+ folder_path

    else:

        output = 'Folder already exit'

    return output

# ---------------------------------------------------
#             返回一个模块的路径
# ---------------------------------------------------

def get_module_path(module):
    module_ini_path = module.__file__
    return get_folder_name_from_path(module_ini_path)


def mymovefile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.move(srcfile,dstfile)          #移动文件
        print ("move %s -> %s"% (srcfile,dstfile))
        
def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%srcfile)
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        print ("copy %s -> %s"%( srcfile,dstfile))
 
def path_uplv(path, lvno):
    folders = path.split('\\')
    new_path = ''
    for i in range(len(folders) - lvno):
        new_path += folders[i] + '\\'
    return new_path
