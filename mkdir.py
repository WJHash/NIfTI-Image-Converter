import os

# 定义根目录
root_dir = "brats"

# 定义子目录结构
sub_dirs = {
    "FLAIR": ["test/image", "test/mask", "train/image", "train/mask"],
    "T1": ["test/image", "test/mask", "train/image", "train/mask"],
    "T1CE": ["test/image", "test/mask", "train/image", "train/mask"],
    "T2": ["test/image", "test/mask", "train/image", "train/mask"]
}

# 创建文件夹结构
for modality, dirs in sub_dirs.items():
    for dir_path in dirs:
        full_path = os.path.join(root_dir, modality, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

print("Folder structure created successfully.")