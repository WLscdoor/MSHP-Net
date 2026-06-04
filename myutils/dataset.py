import os
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import torchvision.transforms as transforms
from PIL import Image
import cv2
from myutils.data_aug import random_augment, add_gaussian_noise

import torch

def binary_threshold_by_max_percent(tensor: torch.Tensor, percent: float) -> torch.Tensor:
    """
    根据张量最大值的百分比进行二值化。元素值 >= (最大值 * percent) 的设为1，否则为0。

    Args:
        tensor (torch.Tensor): 输入张量。
        percent (float): 百分比（0到1之间的数值）。

    Returns:
        torch.Tensor: 二值化后的张量（0或1）。
    """
    max_val = tensor.max()
    threshold = max_val * percent
    return (tensor >= threshold).float()

class HRRPDataset(Dataset):
    def __init__(self, mat_dir, len = 256, data_aug = False, noise_aug = False):
        self.mat_dir = mat_dir
        self.mat_files = [f for f in os.listdir(mat_dir) if f.endswith('.mat')]
        self.len = len
        self.data_aug = data_aug
        self.noise_aug = noise_aug
        # self.labels = {target: i for i, target in enumerate(["BMP2_SN_9563", "BMP2_SN_9566", "BMP2_SN_C21", "BTR70_SN_C71", "T72_SN_132",  "T72_SN_812", "T72_SN_S7"])}
        self.labels = {target: i for i, target in enumerate(["054", "Osman", "Mumbai", "Victory", "Korean",  "DaHai","ESTIMA", "HengHui", "ShunDe", "XinYingKou"])}
        # self.labels = {target: i for i, target in enumerate(["costa", "Boat7", "lng", "patrol","transport3","container1","container2","owlfelino"])}

    def __len__(self):
        
        return len(self.mat_files)


    def __getitem__(self, index):
        mat_file = os.path.join(self.mat_dir, self.mat_files[index])
        data = sio.loadmat(mat_file)
        

        if self.data_aug:
            HRRP_one = torch.tensor(data['Trainingdata']['HRRP_one'][0][0])
            HRRP_one = HRRP_one.numpy()
            HRRP_one = cv2.resize(HRRP_one, (self.len, 1), interpolation=cv2.INTER_CUBIC)
            HRRP_one = torch.tensor(HRRP_one).to(dtype=torch.float32).squeeze(0)
            HRRP_one = random_augment(HRRP_one)
            HRRP_label = binary_threshold_by_max_percent(HRRP_one, 0.01)
            HRRP_one = add_gaussian_noise(HRRP_one, (10,10))
            # HRRP_one = HRRP_one/HRRP_one.max()
            # HRRP_one = HRRP_one.abs()
        else:
            HRRP_one = torch.tensor(data['Trainingdata']['HRRP_one'][0][0])
            HRRP_one = HRRP_one.numpy()
            HRRP_one = cv2.resize(HRRP_one, (self.len, 1), interpolation=cv2.INTER_CUBIC)
            HRRP_one = torch.tensor(HRRP_one).to(dtype=torch.float32)
            HRRP_label = binary_threshold_by_max_percent(HRRP_one, 0.01)
            if self.noise_aug:
                HRRP_one = add_gaussian_noise(HRRP_one, (10,10))
            # HRRP_one = HRRP_one/HRRP_one.max()
            # HRRP_one = HRRP_one.abs()


        label = self.labels[data['Trainingdata']['label'][0][0].item()]
        label = torch.tensor(label, dtype=torch.long)
        return HRRP_one, HRRP_label, label, self.mat_files[index]