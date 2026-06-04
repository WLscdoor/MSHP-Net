import torch
import matplotlib.pyplot as plt

def generate_distance_matrix(N):
    distance_matrix = torch.zeros(N, N, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = 1 / (abs(i - j) + 1)
    return distance_matrix

def generate_distance_matrix_arbitrary(N, M):
    distance_matrix = torch.zeros(N, N, dtype=torch.float32)
    distance_matrix_arb = torch.zeros(M*N, M*N, dtype=torch.float32)
    for i in range(N):
        for j in range(N):
            distance_matrix[i, j] = 1 / (abs(i - j) + 1)

    for i in range(M):
        for j in range(i,M):
            start_i = i * N
            end_i = start_i + N
            start_j = j * N
            end_j = start_j + N
            distance_matrix_arb[start_i:end_i, start_j:end_j] = distance_matrix/(abs(i-j)+1)

    return distance_matrix_arb

# def generate_sinc_torch(
#     length: int,
#     peak_index: int = 0,  # 峰值所在索引（0-based）
#     scale: float = 1.0,
#     device: torch.device = torch.device("cuda")
# ) -> torch.Tensor:
#     """
#     生成归一化 sinc 函数，确保峰值出现在指定索引位置。
    
#     参数：
#     length: int
#         序列长度。
#     peak_index: int (默认0)
#         峰值出现的索引（0-based）。
#     scale: float (默认1.0)
#         控制主瓣宽度（scale越大，主瓣越宽）。
#     device: torch.device (默认cuda)
#         张量所在设备。
    
#     返回：
#     torch.Tensor: 形状为 (length,) 的张量，峰值在 peak_index 处。
#     """
#     # 确保 peak_index 合法
#     assert 0 <= peak_index < length, "peak_index 必须在 [0, length) 范围内"
    
#     # 生成 x 轴，确保 x[peak_index] = 0
#     # x 的范围将根据 peak_index 调整，使 x[peak_index] = 0
#     start = -5.0 * (peak_index / (length - 1))  # 起始点
#     end = 5.0 * ((length - 1 - peak_index) / (length - 1))  # 结束点
#     x = torch.linspace(start, end, length, device=device)
    
#     scaled_x = x / scale  # 因为峰值处 x=0，所以 scaled_x = 0/scale = 0
    
#     # 计算 sinc 函数
#     y = torch.sin(torch.pi * scaled_x) / (torch.pi * scaled_x)
    
#     # 处理 scaled_x 接近 0 的情况（浮点误差）
#     y = torch.where(
#         torch.abs(scaled_x) < 1e-8,
#         torch.tensor(1.0, device=device),
#         y
#     )
    
#     return torch.abs(y)

# def generate_sinc_torch(
#     length: int,
#     peak_index: int = 0,
#     scale: torch.Tensor = torch.tensor(1.0),  # 修改为张量类型
#     device: torch.device = torch.device("cuda")
# ) -> torch.Tensor:
#     """
#     确保 scale 是张量，以保持计算图的连通性。
#     """
#     # 确保 peak_index 合法
#     assert 0 <= peak_index < length, "peak_index 必须在 [0, length) 范围内"
    
#     # 生成 x 轴，确保 x[peak_index] = 0
#     start = -5.0 * (peak_index / (length - 1))
#     end = 5.0 * ((length - 1 - peak_index) / (length - 1))
#     x = torch.linspace(start, end, length, device=device)
    
#     # scaled_x 必须是张量与张量的运算
#     scaled_x = x / scale  # scale 是张量，因此结果也是张量
    
#     # 计算 sinc 函数（保持张量运算）
#     y = torch.sin(torch.pi * scaled_x) / (torch.pi * scaled_x)
    
#     # 处理 scaled_x 接近 0 的情况（使用张量操作）
#     y = torch.where(
#         torch.abs(scaled_x) < 1e-8,
#         torch.ones_like(y, device=device),  # 使用 ones_like 保持设备和类型一致
#         y
#     )
    
    # return torch.abs(y)


def generate_sinc_torch(
    length: int,
    peak_index: int = 0,
    scale: torch.Tensor = torch.tensor(1.0),  # 修改为张量类型
    device: torch.device = torch.device("cuda")
) -> torch.Tensor:
    """
    确保 scale 是张量，以保持计算图的连通性。
    """
    # 确保 peak_index 合法
    assert 0 <= peak_index < length, "peak_index 必须在 [0, length) 范围内"
    
    # 生成 x 轴，确保 x[peak_index] = 0
    start = -5.0 * (peak_index / (length - 1))
    end = 5.0 * ((length - 1 - peak_index) / (length - 1))
    x = torch.linspace(start, end, length, device=device)
    
    # scaled_x 必须是张量与张量的运算
    scaled_x = x / scale  # scale 是张量，因此结果也是张量
    y = torch.ones(length)
    
    # 计算 sinc 函数（保持张量运算）
    if peak_index>0:
        y[0 : peak_index-1] = torch.sin(torch.pi * scaled_x[0 : peak_index-1]) / (torch.pi * scaled_x[0 : peak_index-1])
        y[peak_index + 1 :] = torch.sin(torch.pi * scaled_x[peak_index + 1 :]) / (torch.pi * scaled_x[peak_index + 1 :])
        if peak_index == 1:
            y[0] = torch.sin(torch.pi * scaled_x[0]) / (torch.pi * scaled_x[0])

    if peak_index==0:
        y[peak_index + 1 :] = torch.sin(torch.pi * scaled_x[peak_index + 1 :]) / (torch.pi * scaled_x[peak_index + 1 :])
    
    # if peak_index+1 == length:
    #     y[0: peak_index] = torch.sin(torch.pi * scaled_x[0: peak_index - 1]) / (torch.pi * scaled_x[0: peak_index - 1])
        

    return y
    # return torch.abs(y)


def generate_distance_matrix_sinc(N, scale_tensor):

    distance_matrix = torch.zeros(N, N, dtype=torch.float32, device=scale_tensor.device)
    for i in range(N):
        distance_matrix[i, :] = generate_sinc_torch(
            length=N,
            peak_index=i,
            scale=scale_tensor  # 直接传递张量
        )
    return distance_matrix


# 对N降采样，对M升采样
def generate_distance_matrix_multiscale(N, M, scale=3):
    multiscale_matrix = [None]*scale
    for sc in range(0, scale):
        base_matrix = generate_distance_matrix(int(N/(2**sc)))
        base_N = int(N / (2**sc))
        base_M = int(M / (2**(scale - sc - 1)))
        distance_matrix_multiscale = torch.zeros(int(base_N * base_M), int(base_N * base_M), dtype=torch.float32)
        for i in range(0, M, (2**(scale-sc-1))):
            # for j in range(0,M):
            for j in range(0, M, (2**(scale-sc-1))):
                # if j % (2**(scale-sc-1)) == 0:
                # if j == i:
                    start_i = int(i / (2**(scale-sc-1))* base_N)
                    end_i = start_i + base_N
                    start_j = int(j / (2**(scale-sc-1))* base_N)
                    end_j = start_j + base_N
                    # distance_matrix_multiscale[start_i:end_i, start_j:end_j] = base_matrix/(abs(i-j)+1)
                    distance_matrix_multiscale[start_i:end_i, start_j:end_j] = base_matrix/(abs(int(i / (2**(scale-sc-1)))-int(j / (2**(scale-sc-1))))+1)
        multiscale_matrix[sc] = distance_matrix_multiscale

    return multiscale_matrix