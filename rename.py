import os

def rename_files_in_directory(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件名是否包含 "_0000"
        if "_0000" in filename:
            # 创建新的文件名，去掉 "_0000"
            new_name = filename.replace("_0000", "")
            # 获取文件的完整路径
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            # 重命名文件
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} -> {new_name}')

# 设置文件夹路径
folder_path = "/data1/hf_model/lxdata/Dataset504_IMAGECAS/imagesTs"
rename_files_in_directory(folder_path)
