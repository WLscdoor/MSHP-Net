import torch
import torch.nn.functional as F
import math

def add_gaussian_noise(
    x: torch.Tensor,
    snr_range: tuple = (10, 30),  # SNR范围（单位：dB）
) -> torch.Tensor:
    """
    为1D信号添加随机SNR的高斯白噪声
    Args:
        x: 输入信号 [1, L] 或 [L]
        snr_range: 信噪比范围（dB），如 (10, 30) 表示10dB到30dB
        return_noise: 是否同时返回噪声信号
    Returns:
        noisy_x: 加噪后的信号（形状同输入）
        (可选) noise: 噪声信号
    """
    # 统一输入形状为 [1, L]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2, "输入应为1D信号 [1, L] 或 [L]"
    
    # 随机生成SNR（线性值）
    snr_db = torch.empty(1).uniform_(*snr_range).item()
    snr_linear = 10 ** (snr_db / 10)
    
    # 计算信号功率
    signal_power = torch.mean(x ** 2)
    
    # 根据SNR计算噪声功率
    noise_power = signal_power / snr_linear
    
    # 生成高斯白噪声
    noise = torch.randn_like(x) * math.sqrt(noise_power)
    
    # 添加噪声
    noisy_x = x + noise
    
    return noisy_x

def random_circular_shift(x: torch.Tensor, max_shift: int = 32):
    """
    对1D信号进行随机循环平移（超出部分从另一侧循环）
    Args:
        x: [1, 256] 输入张量
        max_shift: 最大平移量（绝对值）
    Returns:
        shifted_x: [1, 256] 平移后的信号
    """
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    if shift == 0:
        return x
    return torch.roll(x, shifts=shift, dims=1)

def random_zoom(
    x: torch.Tensor, 
    zoom_range: tuple = (0.8, 1.2),
    mode: str = 'linear'
) -> torch.Tensor:
    """
    对任意长度的1D信号进行随机中心缩放（放大或缩小）
    
    Args:
        x: 输入信号，形状为 [1, L] 或 [L]
        zoom_range: 缩放比例范围（如 (0.8, 1.2) 表示缩小到80%或放大到120%）
        mode: 插值模式 ('linear', 'nearest', 'cubic')
        
    Returns:
        zoomed_x: 缩放后的信号，形状与输入相同
    """
    # 统一输入形状为 [1, L]
    if x.dim() == 1:
        x = x.unsqueeze(0)
    assert x.dim() == 2, "输入应为1D信号 [1, L] 或 [L]"
    
    L = x.size(1)
    zoom_factor = torch.empty(1).uniform_(*zoom_range).item()
    if zoom_factor == 1.0:
        return x.clone()
    
    # 计算缩放后的长度
    new_len = int(L * zoom_factor)
    zoomed_x = F.interpolate(x.unsqueeze(1), size=new_len, mode=mode).squeeze(1)
    
    if new_len > L:  # 放大：裁剪中心部分
        start = (new_len - L) // 2
        zoomed_x = zoomed_x[:, start:start+L]
        if zoomed_x.size(1) < L:  # 防越界补零
            zoomed_x = F.pad(zoomed_x, (0, L - zoomed_x.size(1)))
    else:  # 缩小：对称填充
        pad_left = (L - new_len + 1) // 2
        pad_right = L - new_len - pad_left
        zoomed_x = F.pad(zoomed_x, (pad_left, pad_right))
    
    return zoomed_x  # 保持输入形状 [1, L]

def random_shift(x: torch.Tensor, max_shift: int = 32):
    shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
    
    if shift == 0:
        return x
    
    if shift > 0:  # 左移
        shifted_x = torch.cat([x[:, shift:], torch.zeros(1, shift)], dim=1)
    
    if shift < 0:          # 右移
        shifted_x = torch.cat([torch.zeros(1, -shift), x[:, :shift]], dim=1)
    
    return shifted_x

def random_augment(x: torch.Tensor):
    """
    随机选择平移或缩放增广
    Args:
        x: [1, 256] 输入张量
    Returns:
        augmented_x: [1, 256] 增广后的信号
    """
    # if torch.rand(1) < 0.5:
    #     return random_shift(x)
    # else:
    #     return random_zoom(x)
    x = random_zoom(x)
    x = random_shift(x)
    return x



# import matplotlib.pyplot as plt

# # 构造正弦波信号 [1, 100]
# L = 100
# x = torch.sin(torch.linspace(0, 2 * 3.1416, L)).unsqueeze(0)

# # 测试缩小 (zoom_factor=0.7)
# zoomed_out = random_zoom(x, zoom_range=(0.7, 0.7))
# plt.figure(figsize=(12, 4))
# plt.plot(x.squeeze(), label="Original")
# plt.plot(zoomed_out.squeeze(), label="Zoomed x0.7")
# plt.legend()
# plt.title("Zoom Out (factor=0.7)")
# plt.show()

# # 测试放大 (zoom_factor=1.5)
# zoomed_in = random_zoom(x, zoom_range=(1.5, 1.5))
# plt.figure(figsize=(12, 4))
# plt.plot(x.squeeze(), label="Original")
# plt.plot(zoomed_in.squeeze(), label="Zoomed x1.5")
# plt.legend()
# plt.title("Zoom In (factor=1.5)")
# plt.show()

# aa = 1