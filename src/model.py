"""
U-Net with Attention for Face Reconstruction.
Optimized for high-accuracy forensic reconstruction (Inpainting).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# BUILDING BLOCKS
# ============================================================
class ConvBlock(nn.Module):
    """Double convolution block with normalization and activation."""
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.dropout = nn.Dropout2d(0.2) if use_dropout else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x):
        return self.conv(self.pool(x))


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UpBlockWithAttention(nn.Module):
    """Upsampling block with attention mechanism."""
    def __init__(self, in_ch, out_ch, skip_ch, bilinear=True):
        super().__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # in_ch is the total concatenated size (prev_up + skip)
        # We need to derive internal sizes for attention
        # Prev layer (gating) has (in_ch - skip_ch) channels
        gating_ch = in_ch - skip_ch
        
        self.attention = AttentionGate(F_g=gating_ch, F_l=skip_ch, F_int=skip_ch//2)
        self.conv = ConvBlock(in_ch, out_ch)
    
    def forward(self, x1, x2):
        # x1 = Decoding features (Gating signal)
        # x2 = Encoder features (Skip connection)
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x2 = self.attention(g=x1, x=x2)
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


# ============================================================
# U-NET MODEL
# ============================================================
class UNetReconstruction(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Encoder
        self.inc = ConvBlock(in_channels, base_channels)       # 64
        self.down1 = DownBlock(base_channels, base_channels*2) # 128
        self.down2 = DownBlock(base_channels*2, base_channels*4) # 256
        self.down3 = DownBlock(base_channels*4, base_channels*8) # 512
        self.down4 = DownBlock(base_channels*8, base_channels*16) # 1024
        
        # Decoder
        # To fix channel mismatch, we explicitly calculate:
        # Input = Upsampled(Previous) + Skip(Corresponding Encoder)
        
        # Block 1: Up(1024) + Skip(512) = 1536 -> Out 512
        if use_attention:
            self.up1 = UpBlockWithAttention(1536, 512, skip_ch=512)
        else:
            self.up1 = UpBlock(1536, 512)
            
        # Block 2: Up(512) + Skip(256) = 768 -> Out 256
        if use_attention:
            self.up2 = UpBlockWithAttention(768, 256, skip_ch=256)
        else:
            self.up2 = UpBlock(768, 256)
            
        # Block 3: Up(256) + Skip(128) = 384 -> Out 128
        if use_attention:
            self.up3 = UpBlockWithAttention(384, 128, skip_ch=128)
        else:
            self.up3 = UpBlock(384, 128)
            
        # Block 4: Up(128) + Skip(64) = 192 -> Out 64
        if use_attention:
            self.up4 = UpBlockWithAttention(192, 64, skip_ch=64)
        else:
            self.up4 = UpBlock(192, 64)
        
        # Output
        self.outc = nn.Conv2d(64, out_channels, 1)
        self.tanh = nn.Tanh()
        
        self._init_weights()
        logger.info(f"UNetReconstruction initialized (attention={use_attention})")
        logger.info(f"  Parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return self.tanh(logits)

def create_model(model_type='unet_attention', device='cuda', **kwargs):
    if model_type == 'unet':
        model = UNetReconstruction(use_attention=False, **kwargs)
    elif model_type == 'unet_attention':
        model = UNetReconstruction(use_attention=True, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model.to(device)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model('unet_attention', device=device)
    
    x = torch.randn(2, 3, 512, 512).to(device)
    try:
        y = model(x)
        print(f"Input: {x.shape}")
        print(f"Output: {y.shape}")
        
        if y.shape == x.shape:
            print("✓ Model Test Passed")
        else:
            print("❌ Shape Mismatch")
    except Exception as e:
        print(f"❌ Error: {e}")