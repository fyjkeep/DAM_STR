import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedBernoulli

class PostProcessor(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, temp=0.5, kernel_size=5):
        super().__init__()
        
        # 3层1x1卷积
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        
        # Gumbel-Softmax相关
        self.temp = temp  # Gumbel-Softmax温度参数
        
        # 注意力偏移预测
        self.offset_pred = nn.Conv2d(in_channels, 2, kernel_size=3, padding=1)

        # 高斯核
        self.gaussian_kernel = self._create_gaussian_kernel(kernel_size)
        self.gaussian_kernel = self.gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        self.gaussian_kernel = self.gaussian_kernel.to(device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化权重
        self._init_weights()
    
    def _create_gaussian_kernel(self, kernel_size=5, sigma=1.0):
        """创建2D高斯核"""
        ax = torch.arange(kernel_size).float() - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel
    
    def _init_weights(self):
        """初始化权重"""
        for m in [self.conv1, self.conv2, self.conv3, self.offset_pred]:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 1x1卷积处理
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        
        # 保存原始x用于后续处理
        original_x = x
        
        # Gumbel-Softmax处理
        # 首先对特征图进行空间压缩
        batch_size, channels, height, width = x.shape
        x_flat = x.view(batch_size, channels, -1)  # (B, C, H*W)
        
        # Gumbel-Softmax进行离散化
        logits = x_flat.transpose(1, 2)  # (B, H*W, C)
        relaxed_cat = RelaxedBernoulli(temperature=self.temp, logits=logits)
        sample = relaxed_cat.rsample()  # (B, H*W, C)
        
        gumbel_features = sample.transpose(1, 2).view(batch_size, channels, height, width)
        
        # 注意力偏移预测
        offsets = self.offset_pred(original_x)  # (B, 2, H, W)
        offset_x = offsets[:, 0, :, :]  # (B, H, W)
        offset_y = offsets[:, 1, :, :]  # (B, H, W)
        
        # 生成坐标网格
        grid_x, grid_y = torch.meshgrid(torch.arange(width), torch.arange(height))
        grid_x = grid_x.float().to(x.device)
        grid_y = grid_y.float().to(x.device)
        
        offset_grid_x = grid_x + offset_x
        offset_grid_y = grid_y + offset_y
   
        offset_grid_x = 2.0 * offset_grid_x / (width - 1) - 1.0
        offset_grid_y = 2.0 * offset_grid_y / (height - 1) - 1.0
        
        # 采样网格
        sampling_grid = torch.stack([offset_grid_x, offset_grid_y], dim=-1)  # (H, W, 2)
        sampling_grid = sampling_grid.unsqueeze(0)  # (1, H, W, 2)
        sampling_grid = sampling_grid.permute(0, 3, 1, 2)  # (1, 2, H, W)
        sampling_grid = sampling_grid.expand(batch_size, -1, -1, -1)  # (B, 2, H, W)
        
        # 双线性采样
        offset_features = F.grid_sample(original_x, sampling_grid, align_corners=True)
        
        # 高斯模糊处理
        pad = self.gaussian_kernel.shape[-1] // 2
        blurred_features = []
        for c in range(x.shape[1]):
            channel_data = x[:, c, :, :].unsqueeze(1)  # (B, 1, H, W)
            # 使用高斯核进行卷积
            blurred = F.conv2d(channel_data, self.gaussian_kernel, padding=pad)
            blurred_features.append(blurred)
        
        # 合并所有通道
        blurred_features = torch.cat(blurred_features, dim=1)

        final_output = original_x + gumbel_features + offset_features + blurred_features
        
        return final_output


class ResTransformerWithPostProcessing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.res_transformer = ResTranformer(config)
    
        out_channels = 512   
        self.post_processor = PostProcessor(out_channels)
    
    def forward(self, images):
        features = self.res_transformer(images)
        processed_features = self.post_processor(features)
        return processed_features