#!/usr/bin/env python
#########################################
#       nii2png for Python 3.7          #
#         NIfTI Image Converter         #
#                v0.7.1                 #
#                                       #
#     Modified by Assistant             #
#              06 Jan 2025               #
#              MIT License              #
#########################################

import os
import sys
import argparse
import logging
import numpy as np
import nibabel as nib
import imageio
from multiprocessing import Pool
from tqdm import tqdm
from natsort import natsorted  # Imported natsorted for natural sorting

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Convert NIfTI (.nii/.nii.gz) images to PNG format with optional normalization.',
        epilog='Example: nii2png.py -i /path/to/input -o /path/to/output --bit-depth 16 --normalize linear --verbose'
    )
    parser.add_argument('-i', '--input_dir', required=True, 
                        help='输入包含NIfTI文件的目录。')
    parser.add_argument('-o', '--output_dir', required=True, 
                        help='输出PNG图像保存的目录。')
    parser.add_argument('--rotate', type=int, choices=[90, 180, 270], 
                        help='可选的图像旋转角度（90, 180, 或 270 度）。')
    parser.add_argument('--verbose', action='store_true', 
                        help='启用详细日志记录。')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16], default=16,
                        help='输出PNG图像的位深度：8 或 16。默认是16。')
    parser.add_argument('--normalize', type=str, choices=['linear', 'window'], default=None,
                        help='归一化方法： "linear" 进行线性归一化，"window" 进行窗口归一化。默认不进行归一化。')
    parser.add_argument('--window', type=float, default=None,
                        help='窗口宽度（仅在 --normalize window 时使用）。')
    parser.add_argument('--level', type=float, default=None,
                        help='窗口中心（仅在 --normalize window 时使用）。')
    parser.add_argument('--format', type=str, choices=['png', 'jpg', 'tiff'], default='png',
                        help='输出图像的格式。默认是png。')
    return parser.parse_args()

def validate_arguments(args, parser):
    if args.normalize == 'window':
        if args.window is None or args.level is None:
            parser.error('--window and --level are required when using --normalize window')

def setup_logging(verbose: bool):
    log_format = '%(levelname)s: %(message)s'
    if verbose:
        logging.basicConfig(level=logging.INFO, format=log_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=log_format)

def linear_normalize(data: np.ndarray, bit_depth: int) -> np.ndarray:
    """线性归一化数据到0-255或0-65535范围，取决于目标位深度。"""
    data_min = np.min(data)
    data_max = np.max(data)
    logging.info(f'线性归一化 - 数据最小值: {data_min}, 数据最大值: {data_max}')
    if data_max == data_min:
        return np.zeros(data.shape, dtype=np.uint16 if bit_depth == 16 else np.uint8)
    normalized = (data - data_min) / (data_max - data_min)
    if bit_depth == 16:
        scaled = (normalized * 65535).astype(np.uint16)
    else:
        scaled = (normalized * 255).astype(np.uint8)
    return scaled

def window_normalize(data: np.ndarray, window: float, level: float, bit_depth: int) -> np.ndarray:
    """应用窗口归一化并将数据归一化到0-255或0-65535。"""
    lower = level - (window / 2)
    upper = level + (window / 2)
    logging.info(f'窗口归一化 - 窗口宽度: {window}, 窗口中心: {level}, 下限: {lower}, 上限: {upper}')
    clipped = np.clip(data, lower, upper)
    normalized = (clipped - lower) / window
    normalized = np.clip(normalized, 0, 1)
    if bit_depth == 16:
        scaled = (normalized * 65535).astype(np.uint16)
    else:
        scaled = (normalized * 255).astype(np.uint8)
    return scaled

def scale_to_uint16(data: np.ndarray) -> np.ndarray:
    """将浮点数数据缩放到uint16而不进行归一化。"""
    data_min = np.min(data)
    data_max = np.max(data)
    logging.info(f'uint16 缩放 - 数据最小值: {data_min}, 数据最大值: {data_max}')

    if data_max == data_min:
        return np.zeros(data.shape, dtype=np.uint16)
    
    if data_min < 0:
        logging.info('数据包含负值。将数据整体平移以使其非负。')
        data_shifted = data - data_min
    else:
        data_shifted = data

    data_max_shifted = np.max(data_shifted)
    logging.info(f'平移后的数据最大值: {data_max_shifted}')

    if data_max_shifted == 0:
        return np.zeros(data.shape, dtype=np.uint16)
    
    scaled = (data_shifted / data_max_shifted) * 65535  # 16位最大值
    scaled_uint16 = scaled.astype(np.uint16)
    return scaled_uint16

def rotate_image(data: np.ndarray, angle: int) -> np.ndarray:
    """按指定角度旋转图像数据。"""
    k = angle // 90  # 旋转次数
    return np.rot90(data, k)

def process_slice(args):
    slice_data, output_path, rotate_angle, image_format = args
    if rotate_angle:
        slice_data = rotate_image(slice_data, rotate_angle)
    try:
        imageio.imwrite(output_path, slice_data, format=image_format)
        logging.info(f'已保存切片: {output_path}')
        return True
    except Exception as e:
        logging.error(f'保存 {output_path} 失败: {e}')
        return False

def main():
    args = parse_arguments()
    setup_logging(args.verbose)
    validate_arguments(args, sys.argv)

    input_dir = args.input_dir
    output_dir = args.output_dir
    rotate_angle = args.rotate
    bit_depth = args.bit_depth
    normalize_method = args.normalize
    window = args.window
    level = args.level
    image_format = args.format

    logging.info(f'输入目录: {input_dir}')
    logging.info(f'输出目录: {output_dir}')
    if rotate_angle:
        logging.info(f'旋转角度: {rotate_angle} 度')
    else:
        logging.info('未应用旋转。')
    logging.info(f'位深度: {bit_depth}')
    if normalize_method:
        logging.info(f'归一化方法: {normalize_method}')
        if normalize_method == 'window':
            logging.info(f'窗口宽度: {window}, 窗口中心: {level}')
    else:
        logging.info('未进行归一化。')

    if not os.path.isdir(input_dir):
        logging.error(f'输入目录不存在: {input_dir}')
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f'输出目录已设置为: {output_dir}')

    # Collect NIfTI files with natural sorting
    nii_files = [f for f in os.listdir(input_dir) 
                 if f.endswith('.nii') or f.endswith('.nii.gz')]
    nii_files = natsorted(nii_files)  # Apply natural sorting

    if not nii_files:
        logging.error('输入目录中未找到NIfTI文件。')
        sys.exit(1)

    logging.info(f'找到 {len(nii_files)} 个NIfTI文件。')

    tasks = []
    for nii_file in nii_files:
        input_filepath = os.path.join(input_dir, nii_file)
        logging.info(f'正在处理文件: {input_filepath}')

        try:
            img = nib.load(input_filepath)
            data = img.get_fdata()
            logging.info(f'加载 {nii_file} 成功。数据维度: {data.ndim}')
        except Exception as e:
            logging.error(f'加载 {nii_file} 失败: {e}')
            continue

        if data.ndim not in [3, 4]:
            logging.warning(f'跳过 {nii_file}: 不是3D或4D图像。')
            continue

        # Normalize or scale data
        if bit_depth == 16:
            if normalize_method:
                if normalize_method == 'linear':
                    scaled_data = linear_normalize(data, bit_depth)
                elif normalize_method == 'window':
                    try:
                        scaled_data = window_normalize(data, window, level, bit_depth)
                    except ValueError as ve:
                        logging.error(f'窗口归一化参数错误: {ve}')
                        continue
            else:
                scaled_data = scale_to_uint16(data)
        elif bit_depth == 8:
            if normalize_method:
                if normalize_method == 'linear':
                    scaled_data = linear_normalize(data, bit_depth)
                elif normalize_method == 'window':
                    try:
                        scaled_data = window_normalize(data, window, level, bit_depth)
                    except ValueError as ve:
                        logging.error(f'窗口归一化参数错误: {ve}')
                        continue
            else:
                scaled_data = np.clip(data, 0, 255).astype(np.uint8)
        else:
            logging.error(f'不支持的位深度: {bit_depth}')
            sys.exit(1)

        if data.ndim == 3:
            nx, ny, nz = data.shape
            for slice_idx in range(nz):
                slice_data = scaled_data[:, :, slice_idx]
                image_name = f'{os.path.splitext(nii_file)[0]}_slice{slice_idx+1}.{image_format}'
                output_path = os.path.join(output_dir, image_name)
                tasks.append((slice_data, output_path, rotate_angle, image_format))
        elif data.ndim == 4:
            nx, ny, nz, nw = data.shape
            for vol_idx in range(nw):
                for slice_idx in range(nz):
                    slice_data = scaled_data[:, :, slice_idx, vol_idx]
                    image_name = f'{os.path.splitext(nii_file)[0]}_vol{vol_idx+1}_slice{slice_idx+1}.{image_format}'
                    output_path = os.path.join(output_dir, image_name)
                    tasks.append((slice_data, output_path, rotate_angle, image_format))

    if tasks:
        with Pool(processes=os.cpu_count()) as pool:
            # Using imap to preserve the order of tasks
            for _ in tqdm(pool.imap(process_slice, tasks), total=len(tasks), desc='Saving slices'):
                pass  # tqdm handles the progress bar

    logging.info('所有图像转换完成。')

if __name__ == "__main__":
    main()
