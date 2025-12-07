"""
Multi-Task Loss Functions for Forensic Face Reconstruction.
Prioritizes accuracy through pixel (weighted) + perceptual + identity losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. PIXEL-LEVEL RECONSTRUCTION LOSS (Weighted)
# ============================================================
class L1Loss(nn.Module):
    """
    Weighted L1 (MAE) loss.
    Heavily penalizes errors inside the corruption mask (the hole).
    """
    def __init__(self, hole_weight=6.0):
        super().__init__()
        self.hole_weight = hole_weight
        self.loss = nn.L1Loss(reduction='none')  # We handle reduction manually
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: [B, 3, H, W]
            target: [B, 3, H, W]
            mask: [B, 1, H, W] (1.0 = corrupted region, 0.0 = original context)
        """
        # Calculate raw L1 loss map
        loss_map = self.loss(pred, target)  # [B, 3, H, W]
        
        if mask is not None:
            # Expand mask to match channels [B, 1, H, W] -> [B, 3, H, W]
            mask_expanded = mask.expand_as(loss_map)
            
            # Apply weights:
            # Corrupted regions (1.0) -> get hole_weight
            # Context regions (0.0)   -> get 1.0
            weights = 1.0 + (mask_expanded * (self.hole_weight - 1.0))
            
            loss_map = loss_map * weights
            
        return loss_map.mean()

# ============================================================
# 2. PERCEPTUAL LOSS (VGG-based)
# ============================================================
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    Ensures semantic/structural correctness.
    """
    def __init__(self, layers=None, device='cuda'):
        super().__init__()
        
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
        
        self.layers = layers
        self.device = device
        
        # Load VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.to(device).eval()
        
        # Freeze VGG
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.vgg_blocks = nn.ModuleDict()
        layer_map = {'relu1_2': 4, 'relu2_2': 9, 'relu3_3': 16, 'relu4_3': 23}
        
        for name in layers:
            self.vgg_blocks[name] = nn.Sequential(*list(vgg[:layer_map[name]+1]))
        
        # ImageNet Normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        logger.info(f"VGG Perceptual Loss initialized")
    
    def normalize(self, x):
        """Convert [-1, 1] (Data Loader output) to ImageNet Standard."""
        x = (x + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, pred, target):
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        total_loss = 0.0
        for name, block in self.vgg_blocks.items():
            pred_feat = block(pred_norm)
            target_feat = block(target_norm)
            total_loss += F.l1_loss(pred_feat, target_feat)
        
        return total_loss / len(self.vgg_blocks)

# ============================================================
# 3. LPIPS PERCEPTUAL LOSS
# ============================================================
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.warning("LPIPS not installed. Run: pip install lpips")

class LPIPSLoss(nn.Module):
    def __init__(self, net='alex', device='cuda'):
        super().__init__()
        if not LPIPS_AVAILABLE:
            raise ImportError("LPIPS required")
        
        self.model = lpips.LPIPS(net=net).to(device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        logger.info(f"LPIPS Loss initialized ({net})")
    
    def forward(self, pred, target):
        # LPIPS expects [-1, 1], which matches our DataLoader
        return self.model(pred, target).mean()

# ============================================================
# 4. IDENTITY LOSS (FaceNet)
# ============================================================
try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    logger.warning("facenet-pytorch not installed. Run: pip install facenet-pytorch")

class FaceIdentityLoss(nn.Module):
    """
    Ensures the reconstructed face has the same identity as the target.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        if not FACENET_AVAILABLE:
            raise ImportError("facenet-pytorch required")
        
        # Load FaceNet (VGGFace2 weights)
        self.facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()
        for param in self.facenet.parameters():
            param.requires_grad = False
            
        logger.info("FaceNet Identity Loss initialized")
    
    def forward(self, pred, target):
        # Resize to 160x160 (FaceNet requirement)
        # Input is [-1, 1], which works for InceptionResnetV1 standard config
        if pred.shape[-1] != 160:
            pred_rs = F.interpolate(pred, size=(160, 160), mode='bilinear', align_corners=False)
            target_rs = F.interpolate(target, size=(160, 160), mode='bilinear', align_corners=False)
        else:
            pred_rs, target_rs = pred, target
            
        # Extract embeddings
        pred_emb = self.facenet(pred_rs)
        target_emb = self.facenet(target_rs)
        
        # 1 - Cosine Similarity
        cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=1).mean()
        return 1.0 - cos_sim

# ============================================================
# 5. COMBINED LOSS WRAPPER
# ============================================================
class ForensicReconstructionLoss(nn.Module):
    def __init__(
        self, 
        device='cuda',
        w_pixel=1.0, 
        w_perceptual=0.8, 
        w_identity=0.1,  # Keep identity weight low initially to stabilize training
        hole_weight=6.0, # High weight for the corrupted region
        use_lpips=True
    ):
        super().__init__()
        self.device = device
        self.weights = {'pixel': w_pixel, 'perceptual': w_perceptual, 'identity': w_identity}
        
        self.pixel_loss = L1Loss(hole_weight=hole_weight)
        
        if use_lpips and LPIPS_AVAILABLE:
            self.perceptual_loss = LPIPSLoss(device=device)
        else:
            self.perceptual_loss = VGGPerceptualLoss(device=device)
            
        if FACENET_AVAILABLE:
            self.identity_loss = FaceIdentityLoss(device=device)
        else:
            self.identity_loss = None
            
        logger.info(f"Loss Config: {self.weights}, hole_weight={hole_weight}")

    def forward(self, pred, target, mask=None):
        losses = {}
        
        # 1. Pixel Loss (Weighted by mask)
        losses['pixel'] = self.pixel_loss(pred, target, mask)
        
        # 2. Perceptual Loss
        losses['perceptual'] = self.perceptual_loss(pred, target)
        
        # 3. Identity Loss
        if self.identity_loss:
            losses['identity'] = self.identity_loss(pred, target)
        else:
            losses['identity'] = torch.tensor(0.0, device=self.device)
            
        # Total
        losses['total'] = (
            self.weights['pixel'] * losses['pixel'] +
            self.weights['perceptual'] * losses['perceptual'] +
            self.weights['identity'] * losses['identity']
        )
        
        return losses

# TEST SCRIPT
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Dummy Data
    B, C, H, W = 2, 3, 512, 512
    pred = torch.randn(B, C, H, W).to(device)
    target = torch.randn(B, C, H, W).to(device)
    mask = torch.randint(0, 2, (B, 1, H, W)).float().to(device)
    
    print(f"Testing ForensicReconstructionLoss on {device}...")
    criterion = ForensicReconstructionLoss(device=device)
    
    losses = criterion(pred, target, mask)
    
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    
    print("✓ Loss Test Passed")