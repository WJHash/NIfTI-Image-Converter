@echo off
REM 设置 Python 解释器和脚本路径
set PYTHON_SCRIPT=python/nii2png.py

REM 设置根目录
set INPUT_ROOT=F:\allproj\Brain3DVessel\NIfTI-Image-Converter\new_brats
set OUTPUT_ROOT=F:\allproj\Brain3DVessel\NIfTI-Image-Converter\brats

REM 定义模态和子文件夹
set MODALITIES=flair t1 t1ce t2
set SUB_DIRS=test\image test\mask train\image train\mask

REM 遍历所有模态和子文件夹
for %%m in (%MODALITIES%) do (
    for %%d in (%SUB_DIRS%) do (
        set INPUT_DIR=%INPUT_ROOT%\%%m\%%d
        set OUTPUT_DIR=%OUTPUT_ROOT%\%%m\%%d
        echo Processing: %%m - %%d
        python %PYTHON_SCRIPT% -i %INPUT_DIR% -o %OUTPUT_DIR% --bit-depth 16 --rotate 90 --verbose
    )
)

echo All conversions completed!
pause