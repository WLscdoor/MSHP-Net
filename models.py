import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cal_heads(N, scale):
    init_conv = 15
    init_size = 256
    conv_heads = []
    ratio = N / init_size
    result = 2 * math.log2(ratio)
    init_conv = init_conv + result
    for i in range(0, scale):
        conv_heads.append(int((max(9, init_conv-2**(i+1))-1)/2))

    return conv_heads

class basic_conv1d(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(basic_conv1d, self).__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        return x

class var_dwconv1d(nn.Module): 
    def __init__(self, in_ch, out_ch, kernel_size=3, expand = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, in_ch*expand, kernel_size = 1),
            nn.BatchNorm1d(in_ch*expand),            
            nn.GELU(),
            nn.Conv1d(in_ch*expand, in_ch*expand, kernel_size = kernel_size, padding=kernel_size//2, groups=in_ch*expand),
            nn.BatchNorm1d(in_ch*expand),            
            nn.GELU(),
            nn.Conv1d(in_ch*expand, out_ch, kernel_size = 1),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)

        return x

class HybridPerceptionBlock(nn.Module):
    def __init__(self, out_ch, heads, freq_hidden, svd_aug = False):
        super().__init__()

        split = 2
        expand = 2
        self.scale_aware_block = ScaleAwareBlock(out_ch//split, ca_num_heads=heads, svd_aug = svd_aug)
        self.frequency_aware_block = FrequencyAwareBlock(out_ch//split, hidden_dim=freq_hidden)
        self.split_size1 = [(out_ch)//2, (out_ch)//2]
        self.agg_group = nn.Conv1d(out_ch, out_ch * expand, kernel_size=1, groups=out_ch//split, padding=1//2)
        self.norm_group = nn.BatchNorm1d(out_ch * expand)

        self.agg_global = nn.Conv1d(out_ch * expand, out_ch, kernel_size=1)
        self.norm_global = nn.BatchNorm1d(out_ch)

        self.norm1 = nn.BatchNorm1d(out_ch//split)
        self.norm2 = nn.BatchNorm1d(out_ch//split)
        self.norm3 = nn.BatchNorm1d(out_ch * 1)
        self.gelu = nn.GELU()


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, N = x.shape
        x1, x2 = torch.split(x, self.split_size1, dim=1)
        x_mod = self.scale_aware_block(x1) + x1
        x_trans = self.frequency_aware_block(x2) + x2

        x_fus = torch.stack((x_mod, x_trans), dim=2)
        x_fus = x_fus.reshape(B, C, N)
        x_fus = self.norm3(x_fus)
        x_fus = self.gelu(self.norm_group(self.agg_group(x_fus)))
        x_fus = self.norm_global(self.agg_global(x_fus))

        return x_fus

class basic_conv1d_trans(nn.Module):
    def __init__(self, in_ch, out_ch, heads = 1, kernel_size = 3, first_layer = False, block_num = 2, freq_hidden = 64, svd_aug = False):
        super(basic_conv1d_trans, self).__init__()
        self.firstlayer = first_layer
        self.block_num = block_num
        if first_layer:
            self.conv1x1 = PatchEmbedding(out_ch, partition=2, kernel_size = kernel_size)
        self.enc_block = nn.ModuleList()
        for i in range(0, block_num):
            self.enc_block.append(HybridPerceptionBlock(out_ch, heads, freq_hidden, svd_aug=svd_aug))
        

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.firstlayer:
            x = self.conv1x1(x)
        for i in range(0, self.block_num):
            x = self.enc_block[i](x)

        return x

class ScaleAwareBlock(nn.Module):
    def __init__(self, dim, ca_num_heads=4, expand_ratio=2, svd_aug = False):
        super().__init__()

        self.svd_aug = svd_aug
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.act = nn.GELU()
        self.split_groups=self.dim//ca_num_heads
        self.v = nn.Conv1d(dim, dim, kernel_size=1)
        self.s = nn.Conv1d(dim, dim, kernel_size=1)

        # Use different dilation rates to capture local patterns at multiple granularities.
        for i in range(self.ca_num_heads): 
            local_conv = nn.Conv1d(dim, dim, kernel_size=5, padding=(2*(i+1)), stride=1, groups=dim, dilation=i+1)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        
        self.proj0 = nn.Conv1d(dim*ca_num_heads, dim*ca_num_heads, kernel_size=1, padding=0, stride=1, groups=dim)
        self.bn1 = nn.BatchNorm1d(dim*ca_num_heads)
        self.proj1 = nn.Conv1d(dim*ca_num_heads, dim, kernel_size=1, padding=0, stride=1)
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.bn2 = nn.BatchNorm1d(dim)
        if self.svd_aug:
            self.projsvd = nn.Conv1d(dim*ca_num_heads, dim, kernel_size=1, padding=0, stride=1)
            self.bnsvd = nn.BatchNorm1d(dim)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.silu = nn.SiLU()
        self.thres_coef = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def lr_trans(self, x):
        Ut, St, Vth = torch.linalg.svd(x, full_matrices=False)
        thres = torch.sigmoid(self.thres_coef) * St[:,0]
        St = F.relu(St - thres.unsqueeze(-1))
        x_soft = Ut @ torch.diag_embed(St) @ Vth
        return x_soft

    def forward(self, x):
        B, C, N = x.shape

        s_res = self.s(x)
        s = s_res.unsqueeze(0).repeat(self.ca_num_heads, 1, 1, 1)

        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_i = local_conv(s_i)
            s_i = s_i.reshape(B, self.dim, -1, N)
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out,s_i],2)
        s_out = s_out.reshape(B, C*self.ca_num_heads, N)
        
        s_lr = 0
        if self.svd_aug:
            s_lr = self.lr_trans(s_out)
            s_out = s_out-s_lr
            s_lr = self.bnsvd(self.projsvd(s_lr))

        s_out = self.proj0(s_out)
        s_out = self.act(self.bn1(s_out))
        s_out = self.bn2(self.proj1(s_out))
        x = s_out + s_lr

        return x

class FrequencyAwareBlock(nn.Module):
    def __init__(self, dim, hidden_dim = 64):
        super().__init__()
        self.heads = nn.ModuleList()
        hidden_dim = dim * 2
        self.CBAM1 = CBAMLayer(hidden_dim)
        self.thres = nn.Parameter(torch.ones(hidden_dim)*0.1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.modulation1 = nn.Conv1d(dim, hidden_dim, kernel_size=1, padding=0)
        self.modulation2 = nn.Conv1d(dim, hidden_dim, kernel_size=1, padding=0)
        self.modulation3 = nn.Conv1d(dim, hidden_dim, kernel_size=1, padding=0)
        self.agg1 = nn.Conv1d(hidden_dim, dim, kernel_size=1, padding=0)
        self.freqagg = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, padding=0)
        self.gelu = nn.GELU()
        self.silu = nn.SiLU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, _, N = x.shape
        x_glo = self.modulation1(x)
        v = self.modulation2(x)
        x_fft = torch.fft.rfftn(x_glo, norm='ortho', dim=-1)
        mag = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        x_mod_q = self.CBAM1(mag)
        thres = self.relu(self.thres).unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
        thres = thres * self.maxpool(x_mod_q)
        x_mod_q = self.relu(x_mod_q-thres)
        x_fft_out_q = torch.polar(x_mod_q, phase)
        q = torch.fft.irfftn(x_fft_out_q, norm='ortho', dim=-1).transpose(-2, -1)
        k = x_glo.transpose(-2, -1)
        v = v.transpose(-2, -1)
        d_k = q.size(-1)
        attn_scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        x_fft_out = attn_probs @ v
        x_fft_out = x_fft_out.transpose(-2, -1)
        x_fft_out = self.agg1(self.gelu(x_fft_out) * self.gelu(q.transpose(-2, -1)))

        return x_fft_out

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=4, spatial_kernel=1):
        super(CBAMLayer, self).__init__()
 
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
 
        self.mlp = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channel // reduction, channel, 1)
        )

        self.mlp1 = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channel // reduction, channel, 1)
        )
 
        self.conv = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=spatial_kernel, padding=spatial_kernel // 2),
            nn.GELU(),
            nn.Conv1d(4, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2),
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
 
        max_out = self.max_pool(x.transpose(1,2)).transpose(1,2)
        avg_out = self.avg_pool(x.transpose(1,2)).transpose(1,2)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        return x
 
class ECAlayer(nn.Module):
    """Construct an ECA module."""
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, n = x.size()

        y = self.avg_pool(x)

        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        y = self.sigmoid(y)

        return x * y.expand_as(x)

class ClassificationHead(nn.Module):
    def __init__(self, dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)
 
    def forward(self, x):
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_ch, hidden_ch)
        self.fc2 = nn.Linear(hidden_ch, out_ch)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.layernorm1 = nn.LayerNorm(hidden_ch)
        self.layernorm2 = nn.LayerNorm(out_ch)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layernorm1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.layernorm2(x)
        x = self.dropout(x)
        return x

class MultiGatedFusionNetwork(nn.Module):
    def __init__(self, input_size, gate_num):
        super(MultiGatedFusionNetwork, self).__init__()
        self.gate_num = gate_num
        self.receivegate = nn.Sequential(
            nn.LayerNorm(input_size * 2),
            nn.Linear(input_size * 2, input_size),
            nn.LayerNorm(input_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(input_size, 1),
            nn.Sigmoid()
        )
        self.sendgate = nn.Sequential(
            nn.LayerNorm(input_size * (gate_num-1)),
            nn.Linear(input_size * (gate_num-1), input_size),
            nn.LayerNorm(input_size),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(input_size, (gate_num-1)),
            nn.Softmax(dim = -1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, current_input, pre_input, current_scale):

        pre_input = [t for i, t in enumerate(pre_input) if i != current_scale]
        pre_input_gate = torch.cat(pre_input, dim=-1)
        pre_input_gate = self.sendgate(pre_input_gate).unsqueeze(0)
        pre_input_gate = pre_input_gate.transpose(0,-1)
        for i in range(0, self.gate_num-1):
            pre_input[i] = pre_input[i]*pre_input_gate[i]
        pre_input = torch.stack(pre_input, dim=1)
        pre_input = torch.sum(pre_input, dim=1)

        out1 = current_input
        out2 = pre_input

        fused = torch.cat((out1, out2), dim=-1)
        gate = self.receivegate(fused)

        gated_output = (0 + gate) * current_input + (1 - gate) * pre_input
        return gated_output

class graph_fusion_block(nn.Module):
    def __init__(self, current_dim, higher_dim, hidden_dim = 128):
        super(graph_fusion_block, self).__init__()
        self.gate_num = len(higher_dim)
        self.gate = MultiGatedFusionNetwork(hidden_dim, self.gate_num)

    def forward(self, x_highdim, x_lowdim, current_scale):
        
        x_lowdim = x_lowdim.transpose(1,2)
        x_lowdim_store = x_lowdim

        x_lowdim = F.normalize(x_lowdim, p=2, dim=-1)
        x_highdim = [x.clone() for x in x_highdim]
        for i in range(0, self.gate_num):
            if i == current_scale:
                continue
            x_highdim[i] = x_highdim[i].transpose(1,2)
            x_highdim[i] = F.normalize(x_highdim[i], p=2, dim=-1)
            affinity = torch.bmm(x_lowdim, x_highdim[i].transpose(1, 2))
            affinity_matrix = F.softmax(affinity, dim=2)
            x_highdim[i] = torch.bmm(affinity_matrix, x_highdim[i])
        
        fused_features = self.gate(x_lowdim_store, x_highdim, current_scale)
        fused_features = fused_features.transpose(1,2)
    
        return fused_features

class PatchEmbedding(nn.Module):
    def __init__(self, dim, partition = 2, kernel_size = 7):
        super().__init__()
        self.dim = dim
        self.partition = partition
        kernel_size = 13
        expand = 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, dim * expand, kernel_size = kernel_size, padding= kernel_size//2),
            nn.BatchNorm1d(dim * expand),
            nn.LeakyReLU(0.1),
            nn.Conv1d(dim * expand, dim * expand, kernel_size = kernel_size, padding= kernel_size//2),
            nn.BatchNorm1d(dim * expand),
            nn.LeakyReLU(0.1)
        )
        self.conv1 = basic_conv1d(dim * partition * expand, dim, kernel_size = 7)
        self.gelu = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, C, H)
        """
        # Interleave partitions along the channel axis before the final projection.
        x = self.conv(x)
        x_out = []
        for i in range(self.partition):
            x_out.append(x[:,:,i::self.partition])
        x = torch.cat(x_out,dim=1)
        x = self.conv1(x)
        
        return x

class PatchMerging1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv = basic_conv1d(2 * dim, 2 * dim, kernel_size = 5)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: (B, C, H)
        """
        B, C, H = x.shape
        assert H % 2 == 0, f"H size ({H}) must be divisible by 2"
        
        x0 = x[:,:,0::2]
        x1 = x[:,:,1::2]
        x = torch.cat((x0,x1),dim=1)
        
        x = self.conv(x)
        
        return x

class HybridPerceptionEncoder(nn.Module):
    def __init__(self, scale, channel_dim, conv_heads, num_classes, HRRP_len, freq_hidden, block_num):
        super(HybridPerceptionEncoder, self).__init__()
        self.scale = scale
        self.conv1d = nn.ModuleList()
        self.patch_merging = nn.ModuleList()
        self.conv1d.append(basic_conv1d_trans(1, channel_dim[0], heads=conv_heads[0],kernel_size=7, first_layer=True, freq_hidden=freq_hidden, block_num=block_num, svd_aug=True))
        self.patch_merging.append(PatchMerging1D(channel_dim[0]))
        for i in range(0, scale-1):
            svd_aug = False
            self.conv1d.append(basic_conv1d_trans(channel_dim[i], channel_dim[i+1], heads=conv_heads[i], freq_hidden=freq_hidden, block_num=block_num, svd_aug=svd_aug))
            self.patch_merging.append(PatchMerging1D(channel_dim[i]))
        
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
            
        encoder_feature = []
        x_temp = x
        x_temp = self.conv1d[0](x_temp)
        encoder_feature.append(x_temp)

        for i in range(1, self.scale):
            x_temp = self.patch_merging[i](x_temp)
            x_temp  = self.conv1d[i](x_temp) 
            encoder_feature.append(x_temp)     

        return encoder_feature

class LightweightRegionDecoder(nn.Module):
    def __init__(self, scale, channel_dim):
        super(LightweightRegionDecoder, self).__init__()
        self.conv1d = nn.ModuleList()
        self.de_conv = nn.ModuleList()
        self.ds_conv = nn.ModuleList()
        self.de_conv.append(nn.ConvTranspose1d(channel_dim[0]*2, channel_dim[0], kernel_size=3, stride=2, padding=1,output_padding=1))
        self.conv1d.append(var_dwconv1d(channel_dim[0], 1))
        self.ds_conv.append(nn.Conv1d(channel_dim[0], 1, kernel_size=3, padding=1))
        for i in range(1,scale):
            self.de_conv.append(nn.ConvTranspose1d(channel_dim[i]*2, channel_dim[i-1]*2, kernel_size=3, stride=2, padding=1,output_padding=1))
            self.conv1d.append(var_dwconv1d(channel_dim[i], channel_dim[i-1]))
            self.ds_conv.append(nn.Conv1d(channel_dim[i], 1, kernel_size=3, padding=1))
        self.last_fea = var_dwconv1d(channel_dim[-1], channel_dim[-1], expand=1)
        self.scale = scale        

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, encoder_feature):
        
        decoder_feature = [None]*(self.scale-1)
        x_temp = self.last_fea(encoder_feature[-1])
        decoder_feature.append(torch.sigmoid(self.ds_conv[-1](x_temp)))
        for i in range(self.scale-2, -1, -1):
            x_temp = self.de_conv[i](x_temp)
            x_temp = torch.cat((x_temp, encoder_feature[i]), dim=1)
            x_temp = self.conv1d[i+1](x_temp)
            decoder_feature[i] = torch.sigmoid(self.ds_conv[i](x_temp))
            

        return decoder_feature

class HierarchicalCrossScaleAggregationBlock(nn.Module):
    def __init__(self, scale, channel_dim, num_classes, HRRP_len, gate_hidden):
        super(HierarchicalCrossScaleAggregationBlock, self).__init__()
        self.scale = scale
        self.graph_fuser = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(0,scale):
            self.graph_fuser.append(graph_fusion_block(channel_dim[i], channel_dim, hidden_dim = gate_hidden))
            self.conv.append(nn.Sequential(nn.Conv1d(channel_dim[i], gate_hidden, kernel_size=1),
                                    nn.BatchNorm1d(gate_hidden)))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                     nn.Flatten())
        
    def forward(self, x, mask):
            
        label = []
        graph_fea = []
        channel_fea = []
        for i in range(0, self.scale):
            x_temp = x[i]*mask[i]
            x_temp = self.conv[i](x_temp)
            graph_fea.append(x_temp)

        
        for i in range(0, self.scale):
            graph_fusion = self.graph_fuser[i](graph_fea, graph_fea[i], i)
            channel_fea.append(self.avgpool(graph_fusion))


        return label, channel_fea

class MSHPNet(nn.Module):
    def __init__(self, num_classes, emb_dim, N, scale, gate_hidden, freq_hidden, block_num):
        super(MSHPNet,self).__init__()
        channel_dim = []
        conv_heads = cal_heads(N, scale)
        for i in range(0, scale):
            channel_dim.append(emb_dim*(2**i))
        self.hp_encoder = HybridPerceptionEncoder(scale, channel_dim, conv_heads,num_classes,N, freq_hidden, block_num)
        self.hcsa_block = HierarchicalCrossScaleAggregationBlock(scale, channel_dim, num_classes,N, gate_hidden=gate_hidden)
        self.lwr_decoder = LightweightRegionDecoder(scale, channel_dim)
        self.fc = nn.Linear((scale)*gate_hidden, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=1e-6)

    def forward(self, HRRP):

        enc_fea = self.hp_encoder(HRRP)
        dec_fea = self.lwr_decoder(enc_fea)
        hrrp_out, channel_fea = self.hcsa_block(enc_fea, dec_fea)

        class_hrrp = torch.cat(channel_fea, dim=-1)
        
        class_out =self.fc(class_hrrp)
        


        return class_out, hrrp_out, dec_fea, class_hrrp


encoder_block = HybridPerceptionBlock
scale_attention = ScaleAwareBlock
freq_attention = FrequencyAwareBlock
multiscale_HRRPencoder = HybridPerceptionEncoder
multiscale_HRRPdecoder = LightweightRegionDecoder
multiscale_graphencoder = HierarchicalCrossScaleAggregationBlock
HRRP_image_net = MSHPNet
