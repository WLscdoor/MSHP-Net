import numpy as np
import scipy.io as sio
import torch
import matplotlib.pyplot as plt

def save_mat(tensor):


    # 2. 将张量转换为 NumPy 数组
    numpy_array = tensor.cpu().numpy()

# 3. 组织要保存的字典。键为你希望在 .mat 文件中对应的变量名，值为 NumPy 数组
    mat_dict = {
        'my_tensor': numpy_array
    }

# 4. 指定要保存的文件名（可以带路径）
    save_path = 'output_tensor.mat'

# 5. 调用 scipy.io.savemat 保存
    sio.savemat(save_path, mat_dict)

    print(f'已将张量保存到：{save_path}')

def save_fig(
    tensor: torch.Tensor,
    filename: str = "transparent_line_plot.png",
    line_color: str = 'blue',  # 默认颜色为蓝色
    line_width: float = 2.0,   # 新增参数：线条宽度，默认1.0
    dpi: int = 300,
    figure_size: tuple = (8, 4)
):
    """
    将一维张量保存为透明背景的线条图
    
    Args:
        tensor (torch.Tensor): 输入的一维张量（形状为 (N,)）
        filename (str): 输出文件路径（推荐使用 .png 格式）
        line_color (str): 线条颜色（默认蓝色，例如 '#1f77b4' 或 'red'）
        line_width (float): 线条宽度（默认1.0）
        dpi (int): 图像分辨率（默认300）
        figure_size (tuple): 图像尺寸 (width, height)（默认 (8,4)）
    """
    # 确保张量在CPU上并转换为numpy数组
    data = tensor.cpu().numpy()
    
    # 创建透明背景的画布和坐标轴
    fig = plt.figure(figsize=figure_size, dpi=dpi)
    ax = plt.axes([0, 0, 1, 1], frameon=False)  # 去除边框
    
    # 绘制线条（新增 line_width 参数控制宽度）
    ax.plot(data, color=line_color, linewidth=line_width)
    
    # 设置完全透明的背景
    fig.patch.set_visible(False)
    ax.axis('off')         # 隐藏坐标轴
    ax.set_position([0,0,1,1])  # 全屏显示
    
    # 保存为透明背景图片
    plt.savefig(
        filename,
        transparent=True,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)
