import os

# 设置目标文件夹路径
folder_path = '/data1/home/lixiang/COSTA/labels'

# 获取文件夹下所有文件
files = os.listdir(folder_path)

# 过滤出所有 .png 文件
png_files = [f for f in files if f.lower().endswith('.png')]

# 排序文件，确保文件按原来的顺序排序
png_files.sort()

# 重命名文件
for idx, filename in enumerate(png_files, start=1):
    old_file_path = os.path.join(folder_path, filename)
    new_file_name = f"{idx}.png"
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # 重命名操作
    os.rename(old_file_path, new_file_path)
    print(f"Renamed: {old_file_path} -> {new_file_path}")
