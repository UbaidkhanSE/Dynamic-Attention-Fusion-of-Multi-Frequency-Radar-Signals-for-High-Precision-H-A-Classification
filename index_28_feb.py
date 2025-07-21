import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import copy
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import math

# Constants
NUM_CLASSES = 11
FEATURE_DIM = 256

#######################
### DATA PROCESSING ###
#######################

class RadarSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, radars=['24GHz', '77GHz', 'Xethru'], 
                 apply_augmentation=False, split='train', train_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform
        self.radars = radars
        self.apply_augmentation = apply_augmentation
        self.split = split
        
        # Get activity classes (assuming directories are named by activity)
        self.activities = sorted([d for d in os.listdir(os.path.join(root_dir, radars[0])) 
                                if os.path.isdir(os.path.join(root_dir, radars[0], d))])
        
        # Build the dataset
        self.samples = []
        self.class_counts = {activity: 0 for activity in self.activities}
        
        for activity_idx, activity in enumerate(self.activities):
            sample_files = sorted(os.listdir(os.path.join(root_dir, radars[0], activity)))
            
            # Count valid samples (those that exist across all radar types)
            valid_samples = []
            for sample in sample_files:
                valid = True
                for radar in radars:
                    if not os.path.exists(os.path.join(root_dir, radar, activity, sample)):
                        valid = False
                        break
                
                if valid:
                    valid_samples.append(sample)
                    self.class_counts[activity] += 1
            
            # Split into train/test
            np.random.seed(42)  # For reproducibility
            np.random.shuffle(valid_samples)
            
            split_idx = int(len(valid_samples) * train_ratio)
            
            if split == 'train':
                samples_to_use = valid_samples[:split_idx]
            else:  # Test or val
                samples_to_use = valid_samples[split_idx:]
            
            for sample in samples_to_use:
                self.samples.append((activity, activity_idx, sample))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        activity, label, sample = self.samples[idx]
        
        # Load spectrograms from all radars
        radar_spectrograms = []
        for radar in self.radars:
            img_path = os.path.join(self.root_dir, radar, activity, sample)
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # Apply data augmentation if enabled (for training)
            if self.apply_augmentation:
                image = self.apply_spectrogram_augmentation(image)
            
            radar_spectrograms.append(image)
        
        return radar_spectrograms, label
    
    def apply_spectrogram_augmentation(self, spectrogram):
        """Apply data augmentation to spectrograms"""
        # Random time and frequency masking
        if torch.rand(1).item() < 0.5:
            # Apply time masking
            t = int(spectrogram.shape[2] * torch.rand(1).item() * 0.2)  # Mask up to 20% of time steps
            t0 = torch.randint(0, spectrogram.shape[2] - t + 1, (1,)).item()
            spectrogram[:, :, t0:t0+t] = 0
        
        if torch.rand(1).item() < 0.5:
            # Apply frequency masking
            f = int(spectrogram.shape[1] * torch.rand(1).item() * 0.2)  # Mask up to 20% of frequency bins
            f0 = torch.randint(0, spectrogram.shape[1] - f + 1, (1,)).item()
            spectrogram[:, f0:f0+f, :] = 0
            
        # Add small random noise
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(spectrogram) * 0.05 * torch.rand(1).item()
            spectrogram = spectrogram + noise
            spectrogram = torch.clamp(spectrogram, 0, 1)
        
        # Random time stretching/shrinking
        if torch.rand(1).item() < 0.3:
            orig_size = spectrogram.shape[2]
            stretch_factor = 0.8 + 0.4 * torch.rand(1).item()  # 0.8x to 1.2x
            new_size = int(orig_size * stretch_factor)
            if new_size != orig_size:
                spectrogram = torch.nn.functional.interpolate(
                    spectrogram.unsqueeze(0), size=(spectrogram.shape[1], new_size),
                    mode='bilinear', align_corners=False
                ).squeeze(0)
                # Pad or crop to original size
                if new_size < orig_size:
                    padding = orig_size - new_size
                    spectrogram = torch.nn.functional.pad(
                        spectrogram, (0, padding, 0, 0), mode='constant', value=0
                    )
                else:
                    spectrogram = spectrogram[:, :, :orig_size]
            
        return spectrogram

    def get_class_weights(self):
        """Calculate inverse class weights for handling imbalance"""
        total_samples = sum(self.class_counts.values())
        num_classes = len(self.activities)
        
        # Calculate inverse frequency
        weights = []
        for activity in self.activities:
            if self.class_counts[activity] > 0:
                weight = total_samples / (num_classes * self.class_counts[activity])
            else:
                weight = 1.0  # Default for empty classes
            weights.append(weight)
            
        return torch.FloatTensor(weights)

#######################
### MODEL COMPONENTS ##
#######################

class DenoisingLayer(nn.Module):
    """Layer that applies learnable denoising"""
    def __init__(self, channels=3):
        super().__init__()
        
        # Learnable parameters for filtering
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # Convolutional filters for denoising
        self.denoise_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        # Apply denoising convolution
        denoised = self.denoise_conv(x)
        denoised = self.bn(denoised)
        
        # Mix original and denoised based on learnable parameter
        alpha = torch.sigmoid(self.alpha)
        return alpha * x + (1 - alpha) * denoised


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MultiscaleBlock(nn.Module):
    """Extract features at multiple scales"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        mid_channels = out_channels // 4
        
        # Different kernel sizes for multi-scale processing
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU()
        )
        
        # Channel attention after concatenation
        self.se = SEBlock(out_channels)
        
        # Projection for residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        
        # Final activation
        self.activation = nn.SiLU()
    
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        y1 = self.branch1(x)
        y2 = self.branch2(x)
        y3 = self.branch3(x)
        y4 = self.branch4(x)
        
        y = torch.cat([y1, y2, y3, y4], dim=1)
        y = self.se(y)
        y = y + shortcut
        return self.activation(y)


class TemporalAttentionModule(nn.Module):
    """Module that focuses on temporal patterns in spectrograms"""
    def __init__(self, channels):
        super().__init__()
        
        # Temporal convolution with large kernel in time dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 9), padding=(0, 4)),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # Channel attention
        self.channel_attn = SEBlock(channels)
    
    def forward(self, x):
        # Apply temporal convolution
        temporal = self.temporal_conv(x)
        # Apply channel attention
        output = self.channel_attn(temporal)
        # Residual connection
        return output + x


class FrequencyAttentionModule(nn.Module):
    """Module that focuses on frequency patterns in spectrograms"""
    def __init__(self, channels):
        super().__init__()
        
        # Frequency convolution with large kernel in frequency dimension
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # Channel attention
        self.channel_attn = SEBlock(channels)
    
    def forward(self, x):
        # Apply frequency convolution
        freq = self.freq_conv(x)
        # Apply channel attention
        output = self.channel_attn(freq)
        # Residual connection
        return output + x


class SpatialAttentionModule(nn.Module):
    """Module that focuses on spatial patterns"""
    def __init__(self, channels):
        super().__init__()
        
        # Spatial attention
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Calculate spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv1(attention))
        
        # Apply spatial attention
        return x * attention


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficient processing"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


###########################
### FEATURE EXTRACTORS ####
###########################

class RadarFeatureExtractor(nn.Module):
    def __init__(self, radar_type, input_channels=3):
        super().__init__()
        self.radar_type = radar_type
        
        # Common preprocessing
        self.denoising = DenoisingLayer(input_channels)
        
        # Radar-specific architectures
        if radar_type == '24GHz':
            # Specialized for temporal resolution and low frequency
            self.encoder = nn.Sequential(
                # Initial convolution with larger time dimension receptive field
                nn.Conv2d(input_channels, 32, kernel_size=(3, 7), stride=2, padding=(1, 3)),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Multi-scale feature extraction
                MultiscaleBlock(32, 64),
                MultiscaleBlock(64, 128),
                
                # Temporal attention module
                TemporalAttentionModule(128),
                
                # Final features
                nn.Conv2d(128, FEATURE_DIM, kernel_size=1),
                nn.BatchNorm2d(FEATURE_DIM),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
        elif radar_type == '77GHz':
            # Specialized for spatial resolution and high frequency
            self.encoder = nn.Sequential(
                # Initial convolution optimized for spatial patterns
                nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Multi-scale feature extraction
                MultiscaleBlock(32, 64),
                MultiscaleBlock(64, 128),
                
                # Frequency attention module
                FrequencyAttentionModule(128),
                
                # Final features
                nn.Conv2d(128, FEATURE_DIM, kernel_size=1),
                nn.BatchNorm2d(FEATURE_DIM),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            
        else:  # Xethru
            # Specialized for penetration characteristics
            self.encoder = nn.Sequential(
                # Initial depth-separable convolution for efficiency
                DepthwiseSeparableConv(input_channels, 32, kernel_size=3),
                nn.BatchNorm2d(32),
                nn.SiLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Efficient blocks with SE attention
                MultiscaleBlock(32, 64),
                MultiscaleBlock(64, 128),
                
                # Spatial attention for motion detection
                SpatialAttentionModule(128),
                
                # Final features
                nn.Conv2d(128, FEATURE_DIM, kernel_size=1),
                nn.BatchNorm2d(FEATURE_DIM),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            )
    
    def forward(self, x):
        # Apply denoising
        x = self.denoising(x)
        
        # Extract features
        features = self.encoder(x)
        return features.view(-1, FEATURE_DIM)


###########################
#### FUSION NETWORKS ######
###########################

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion mechanism to combine information from multiple radars"""
    def __init__(self, feature_dim=FEATURE_DIM, num_heads=8):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Queries, keys, and values projections for each radar
        self.q_24g = nn.Linear(feature_dim, feature_dim)
        self.k_24g = nn.Linear(feature_dim, feature_dim)
        self.v_24g = nn.Linear(feature_dim, feature_dim)
        
        self.q_77g = nn.Linear(feature_dim, feature_dim)
        self.k_77g = nn.Linear(feature_dim, feature_dim)
        self.v_77g = nn.Linear(feature_dim, feature_dim)
        
        self.q_xethru = nn.Linear(feature_dim, feature_dim)
        self.k_xethru = nn.Linear(feature_dim, feature_dim)
        self.v_xethru = nn.Linear(feature_dim, feature_dim)
        
        # Output projections
        self.out_projection = nn.Linear(feature_dim, feature_dim)
        
        # Radar importance weights (learned during training)
        self.radar_weights = nn.Parameter(torch.ones(3))
        
        # Layer normalization
        self.norm = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(0.1)
        )
        self.final_norm = nn.LayerNorm(feature_dim)
        
    def get_attention_weights(self):
        """Return normalized radar weights"""
        return F.softmax(self.radar_weights, dim=0)
    
    def forward(self, f24, f77, fx):
        batch_size = f24.size(0)
        
        # Residual connection
        residual = (f24 + f77 + fx) / 3
        
        # Layer normalization
        f24 = self.norm(f24)
        f77 = self.norm(f77)
        fx = self.norm(fx)
        
        # Project queries, keys, and values
        q24 = self.q_24g(f24).view(batch_size, self.num_heads, self.head_dim)
        k24 = self.k_24g(f24).view(batch_size, self.num_heads, self.head_dim)
        v24 = self.v_24g(f24).view(batch_size, self.num_heads, self.head_dim)
        
        q77 = self.q_77g(f77).view(batch_size, self.num_heads, self.head_dim)
        k77 = self.k_77g(f77).view(batch_size, self.num_heads, self.head_dim)
        v77 = self.v_77g(f77).view(batch_size, self.num_heads, self.head_dim)
        
        qx = self.q_xethru(fx).view(batch_size, self.num_heads, self.head_dim)
        kx = self.k_xethru(fx).view(batch_size, self.num_heads, self.head_dim)
        vx = self.v_xethru(fx).view(batch_size, self.num_heads, self.head_dim)
        
        # Calculate attention scores (each radar attends to all others)
        scale = self.head_dim ** -0.5
        
        # 24GHz attending to others
        attn_24_77 = torch.bmm(q24, k77.transpose(1, 2)) * scale
        attn_24_x = torch.bmm(q24, kx.transpose(1, 2)) * scale
        attn_24 = F.softmax(torch.stack([attn_24_77, attn_24_x], dim=1), dim=1)
        
        # 77GHz attending to others
        attn_77_24 = torch.bmm(q77, k24.transpose(1, 2)) * scale
        attn_77_x = torch.bmm(q77, kx.transpose(1, 2)) * scale
        attn_77 = F.softmax(torch.stack([attn_77_24, attn_77_x], dim=1), dim=1)
        
        # Xethru attending to others
        attn_x_24 = torch.bmm(qx, k24.transpose(1, 2)) * scale
        attn_x_77 = torch.bmm(qx, k77.transpose(1, 2)) * scale
        attn_x = F.softmax(torch.stack([attn_x_24, attn_x_77], dim=1), dim=1)
        
        # Apply attention to values
        out_24 = torch.bmm(attn_24[:, 0], v77) + torch.bmm(attn_24[:, 1], vx)
        out_77 = torch.bmm(attn_77[:, 0], v24) + torch.bmm(attn_77[:, 1], vx)
        out_x = torch.bmm(attn_x[:, 0], v24) + torch.bmm(attn_x[:, 1], v77)
        
        # Reshape and apply radar weights
        out_24 = out_24.view(batch_size, self.feature_dim)
        out_77 = out_77.view(batch_size, self.feature_dim)
        out_x = out_x.view(batch_size, self.feature_dim)
        
        # Get normalized radar weights
        radar_weights = F.softmax(self.radar_weights, dim=0)
        
        # Weighted sum
        fused = (
            radar_weights[0] * out_24 + 
            radar_weights[1] * out_77 + 
            radar_weights[2] * out_x
        )
        
        # Output projection
        fused = self.out_projection(fused)
        
        # Residual connection
        fused = fused + residual
        
        # Feed-forward network
        ff_output = self.ff_network(fused)
        
        # Final residual connection and normalization
        output = self.final_norm(fused + ff_output)
        
        return output, radar_weights


class AdaptiveClassifier(nn.Module):
    """Classifier with dropout and adaptive regularization"""
    def __init__(self, feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, dropout_rate=0.5):
        super().__init__()
        
        # Classifier network with progressive narrowing
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Auxiliary classifiers for each radar (for deep supervision)
        self.aux_classifier_24g = nn.Linear(feature_dim, num_classes)
        self.aux_classifier_77g = nn.Linear(feature_dim, num_classes)
        self.aux_classifier_xethru = nn.Linear(feature_dim, num_classes)
        
        # Monte Carlo dropout for uncertainty estimation
        self.mc_dropout = nn.Dropout(dropout_rate)
        self.enable_mc_dropout = False
    
    def forward(self, x, f24=None, f77=None, fx=None):
        # Apply Monte Carlo dropout if enabled
        if self.enable_mc_dropout and self.training:
            x = self.mc_dropout(x)
        
        # Main classifier
        main_output = self.classifier(x)
        
        # Apply auxiliary classifiers if individual features are provided
        aux_outputs = None
        if f24 is not None and f77 is not None and fx is not None:
            aux_out_24g = self.aux_classifier_24g(f24)
            aux_out_77g = self.aux_classifier_77g(f77)
            aux_out_xethru = self.aux_classifier_xethru(fx)
            aux_outputs = [aux_out_24g, aux_out_77g, aux_out_xethru]
        
        return main_output, aux_outputs
    
    def enable_uncertainty(self, enable=True):
        """Enable Monte Carlo dropout for uncertainty estimation"""
        self.enable_mc_dropout = enable


###########################
##### MAIN MODEL ##########
###########################

class HybridMultiRadarNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM):
        super().__init__()
        
        # Specialized feature extractors for each radar type
        self.radar_24g_encoder = RadarFeatureExtractor('24GHz')
        self.radar_77g_encoder = RadarFeatureExtractor('77GHz')
        self.xethru_encoder = RadarFeatureExtractor('Xethru')
        
        # Cross-attention fusion with learned weighting
        self.fusion = CrossAttentionFusion(feature_dim=feature_dim)
        
        # Adaptive classifier with auxiliary outputs
        self.classifier = AdaptiveClassifier(feature_dim=feature_dim, num_classes=num_classes)
        
        # Progressive SNR adaptation parameters
        self.snr_level = nn.Parameter(torch.tensor(30.0), requires_grad=False)  # Starting SNR
        
    def forward(self, inputs, apply_noise=False):
        r24, r77, xethru = inputs
        
        # Apply noise if training with progressive SNR adaptation
        if apply_noise and self.training:
            r24 = self._add_noise(r24, self.snr_level)
            r77 = self._add_noise(r77, self.snr_level)
            xethru = self._add_noise(xethru, self.snr_level)
        
        # Extract features from each radar
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Fusion with cross-attention
        fused, _ = self.fusion(f24, f77, fx)
        
        # Classification
        main_output, aux_outputs = self.classifier(fused, f24, f77, fx)
        
        if self.training:
            return main_output, aux_outputs
        else:
            return main_output
    
    def get_attention_weights(self):
        """Get radar importance weights from fusion module"""
        return self.fusion.get_attention_weights()
    
    def _add_noise(self, signal, snr_db):
        """Add noise to signal based on SNR level"""
        # Calculate signal power
        signal_power = torch.mean(signal**2)
        
        # Calculate noise power from SNR
        noise_power = signal_power / (10**(snr_db/10))
        
        # Generate noise
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        
        return signal + noise
    
    def update_snr_level(self, new_snr):
        """Update SNR level for progressive adaptation"""
        self.snr_level.data = torch.tensor(new_snr)


#######################
##### LOSS FUNCTION ###
#######################

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


#######################
#### TRAINING LOOP ####
#######################

def train_with_progressive_snr(model, train_loader, val_loader, class_weights, num_epochs=2, patience=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use focal loss for better handling of class imbalance
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    
    # Cross-entropy for auxiliary classifiers
    aux_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler with warmup and cosine annealing
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.001, 
        steps_per_epoch=len(train_loader), 
        epochs=num_epochs,
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos'
    )
    
    # For early stopping
    best_val_acc = 0.0
    patience_counter = 0
    
    # For tracking metrics
    # For tracking metrics
    history = {
        'train_loss': [], 'val_loss': [], 
        'train_acc': [], 'val_acc': [],
        'snr_levels': []
    }
    
    # SNR progression schedule (start high, gradually decrease)
    snr_schedule = [30, 20, 15, 10, 5, 3, 0, -3, -5]
    snr_epochs = [0] + [int(num_epochs * (i+1) / len(snr_schedule)) for i in range(len(snr_schedule)-1)]
    current_snr_idx = 0
    
    for epoch in range(num_epochs):
        # Update SNR level according to schedule
        if epoch >= snr_epochs[current_snr_idx] and current_snr_idx < len(snr_schedule) - 1:
            current_snr_idx += 1
            model.update_snr_level(snr_schedule[current_snr_idx])
            print(f"Epoch {epoch}: SNR level updated to {snr_schedule[current_snr_idx]} dB")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with noise
            main_outputs, aux_outputs = model([r24, r77, xethru], apply_noise=True)
            
            # Calculate main loss
            main_loss = criterion(main_outputs, labels)
            
            # Calculate auxiliary losses
            aux_loss = 0
            if aux_outputs is not None:
                for aux_out in aux_outputs:
                    aux_loss += aux_criterion(aux_out, labels)
                aux_loss /= len(aux_outputs)
            
            # Total loss with auxiliary loss weighted at 0.3
            loss = main_loss + 0.3 * aux_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(main_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate epoch metrics
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['snr_levels'].append(model.snr_level.item())
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                r24, r77, xethru = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # Forward pass without noise
                outputs = model([r24, r77, xethru], apply_noise=False)
                
                # If we're in validation mode, outputs is just the main output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                
                # Update statistics
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate epoch metrics
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, '
              f'SNR: {model.snr_level.item():.1f} dB')
        
        # Check class-wise performance
        class_report = classification_report(
            all_labels, all_preds, output_dict=True, 
            zero_division=0
        )
        
        # Early stopping with patience
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), './best_hybrid_model.pth')
            patience_counter = 0
            
            # Report per-class performance for best model
            print("Class-wise performance:")
            for i, class_name in enumerate(val_loader.dataset.activities):
                if str(i) in class_report:
                    metrics = class_report[str(i)]
                    print(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                          f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return history


#######################
#### EVALUATION #######
#######################

def evaluate_model(model, test_loader, class_names):
    """Comprehensive evaluation of the model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Get model outputs
            outputs = model([r24, r77, xethru], apply_noise=False)
            
            # If we're in evaluation mode, outputs is just the main output
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Calculate precision, recall, F1 for each class
    precision = [report[class_name]['precision'] for class_name in class_names]
    recall = [report[class_name]['recall'] for class_name in class_names]
    f1 = [report[class_name]['f1-score'] for class_name in class_names]
    
    # Calculate prediction certainty (max probability)
    all_probs = np.array(all_probs)
    certainty = np.max(all_probs, axis=1)
    
    # Get radar importance weights
    radar_weights = model.get_attention_weights().cpu().detach().numpy()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'certainty': certainty,
        'radar_weights': radar_weights
    }


def uncertainty_estimation(model, test_loader, num_samples=30):
    """Estimate prediction uncertainty using Monte Carlo Dropout"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()  # Set to train mode to enable dropout
    model.classifier.enable_uncertainty(True)
    
    all_labels = []
    all_predictions = []
    sample_predictions = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            all_labels.extend(labels.numpy())
            
            # Collect multiple predictions with dropout enabled
            batch_samples = []
            for _ in range(num_samples):
                outputs = model([r24, r77, xethru])
                
                # If we're in training mode with uncertainty, outputs is a tuple
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                probs = F.softmax(outputs, dim=1)
                batch_samples.append(probs.cpu().numpy())
            
            # Average predictions
            mean_probs = np.mean(np.array(batch_samples), axis=0)
            sample_predictions.append(np.array(batch_samples))
            
            # Get final predictions
            preds = np.argmax(mean_probs, axis=1)
            all_predictions.extend(preds)
    
    # Calculate entropy as a measure of uncertainty
    stacked_samples = np.vstack(sample_predictions)
    mean_probs = np.mean(stacked_samples, axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Disable uncertainty estimation
    model.classifier.enable_uncertainty(False)
    model.eval()
    
    return {
        'accuracy': accuracy,
        'entropy': entropy,
        'mean_entropy': np.mean(entropy)
    }


def simulate_communication_constraints(model, val_loader, snr_levels=[-10, -5, 0, 5, 10, 20, 30]):
    """Simulate performance under different SNR conditions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for snr in snr_levels:
        # Update model's SNR level
        model.update_snr_level(snr)
        
        # Evaluate
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to device
                r24, r77, xethru = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # Forward pass with noise
                outputs = model([r24, r77, xethru], apply_noise=True)
                
                # If we're in a mode that returns aux outputs
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        if total > 0:
            results[snr] = correct / total
    
    return results


def perform_cross_validation(model_class, dataset, class_weights, k=5, num_epochs=20):
    """Perform k-fold cross-validation"""
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # For storing results
    fold_results = []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get indices
    indices = list(range(len(dataset)))
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"FOLD {fold+1}/{k}")
        print("-" * 30)
        
        # Create data samplers
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        # Create dataloaders
        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=0)
        
        # Initialize model
        model = model_class(num_classes=NUM_CLASSES)
        model.to(device)
        
        # Define loss and optimizer
        criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
        aux_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                # Move to device
                radar_inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                main_outputs, aux_outputs = model(radar_inputs, apply_noise=True)
                
                # Calculate main loss
                main_loss = criterion(main_outputs, labels)
                
                # Calculate auxiliary losses
                aux_loss = 0
                if aux_outputs is not None:
                    for aux_out in aux_outputs:
                        aux_loss += aux_criterion(aux_out, labels)
                    aux_loss /= len(aux_outputs)
                
                # Total loss with auxiliary loss weighted at 0.3
                loss = main_loss + 0.3 * aux_loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(main_outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total if train_total > 0 else 0
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # Move to device
                    radar_inputs = [inp.to(device) for inp in inputs]
                    labels = labels.to(device)
                    
                    # Forward pass without noise
                    outputs = model(radar_inputs, apply_noise=False)
                    
                    # If in evaluation mode, outputs is just the main output
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Loss
                    loss = criterion(outputs, labels)
                    
                    # Statistics
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Final validation accuracy for this fold
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                radar_inputs = [inp.to(device) for inp in inputs]
                labels = labels.to(device)
                outputs = model(radar_inputs, apply_noise=False)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        fold_acc = correct / total if total > 0 else 0
        fold_results.append(fold_acc)
        print(f"Fold {fold+1} accuracy: {fold_acc:.4f}")
    
    # Calculate average results
    avg_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f"Cross-validation complete!")
    print(f"Average accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
    
    return fold_results, avg_acc, std_acc


#######################
#### VISUALIZATION ####
#######################

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot SNR progression
    plt.subplot(1, 3, 3)
    plt.plot(history['snr_levels'], marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('SNR (dB)')
    plt.title('SNR Progression During Training')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_radar_importance(radar_weights, class_names=None, class_values=None):
    """Plot radar importance for different classes"""
    radar_types = ['24GHz', '77GHz', 'Xethru']
    
    plt.figure(figsize=(10, 6))
    
    if class_values is not None and class_names is not None:
        # Plot per-class radar importance
        x = np.arange(len(class_names))
        width = 0.25
        
        for i, radar in enumerate(radar_types):
            plt.bar(x + i*width - width, class_values[radar], width, label=radar)
        
        plt.xlabel('Activity Class')
        plt.ylabel('Radar Importance')
        plt.title('Radar Importance by Activity Class')
        plt.xticks(x, class_names, rotation=45, ha='right')
    else:
        # Plot overall radar importance
        plt.bar(radar_types, radar_weights)
        plt.ylabel('Importance Weight')
        plt.title('Overall Radar Importance')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('radar_importance.png')
    plt.show()


def plot_tsne_visualization(model, val_loader):
    """Visualize feature space using t-SNE"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    features = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Extract features from each encoder
            f24 = model.radar_24g_encoder(r24)
            f77 = model.radar_77g_encoder(r77)
            fx = model.xethru_encoder(xethru)
            
            # Get fused features
            fused, _ = model.fusion(f24, f77, fx)
            
            features.append(fused.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.vstack(features)
    labels_array = np.concatenate(labels_list)
    
    # Apply t-SNE
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, verbose=1)
    features_tsne = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 10))
    for i, activity in enumerate(val_loader.dataset.activities):
        mask = labels_array == i
        plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=activity)
    
    plt.legend()
    plt.title('t-SNE Visualization of Fused Features')
    plt.savefig('tsne_features.png')
    plt.show()


def plot_snr_vs_accuracy(snr_results):
    """Plot performance vs SNR level"""
    plt.figure(figsize=(10, 6))
    
    snrs = list(snr_results.keys())
    accuracies = [snr_results[s] for s in snrs]
    
    plt.plot(snrs, accuracies, marker='o', linestyle='-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('Performance under Different Noise Levels')
    plt.grid(True)
    
    # Add data labels
    for x, y in zip(snrs, accuracies):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    plt.savefig('snr_vs_accuracy.png')
    plt.show()


#######################
###### MAIN SCRIPT ####
#######################

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RadarSpectrogramDataset(
        root_dir="E:\\SpectrogramFusion\\11_class_activity_data",  # Update with actual path
        transform=transform,
        apply_augmentation=True,
        split='train'
    )
    
    val_dataset = RadarSpectrogramDataset(
        root_dir="E:\\SpectrogramFusion\\11_class_activity_data.rar",  # Update with actual path
        transform=transform,
        apply_augmentation=False,
        split='val'
    )
    
    # Get class weights for handling imbalance
    class_weights = train_dataset.get_class_weights()
    class_names = train_dataset.activities
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Initialize model
    model = HybridMultiRadarNet(num_classes=len(class_names))
    
    # Print model architecture overview
    print("Model Architecture:")
    print(f"- Feature Extractors: Specialized CNNs for each radar modality")
    print(f"- Fusion Mechanism: Cross-attention with adaptive weighting")
    print(f"- Classifier: Multi-layer with auxiliary outputs")
    print(f"- Training Strategy: Progressive SNR adaptation, focal loss, stratified CV")
    
    # Perform cross-validation first
    print("\nPerforming cross-validation...")
    cv_results, cv_avg, cv_std = perform_cross_validation(
        HybridMultiRadarNet, train_dataset, class_weights, k=5, num_epochs=10
    )
    
    # Train full model with progressive SNR adaptation
    print("\nTraining full model with progressive SNR adaptation...")
    history = train_with_progressive_snr(
        model, train_loader, val_loader, class_weights, num_epochs=100
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    best_model = HybridMultiRadarNet(num_classes=len(class_names))
    best_model.load_state_dict(torch.load('./best_hybrid_model.pth'))
    
    # Comprehensive evaluation
    print("\nEvaluating best model...")
    results = evaluate_model(best_model, val_loader, class_names)
    
    # Print evaluation results
    print(f"\nOverall Accuracy: {results['accuracy']:.4f}")
    
    print("\nClass-wise Performance:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={results['precision'][i]:.4f}, "
              f"Recall={results['recall'][i]:.4f}, F1={results['f1'][i]:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Plot radar importance
    plot_radar_importance(results['radar_weights'])
    
    # Visualize feature space
    plot_tsne_visualization(best_model, val_loader)
    
    # Perform uncertainty estimation
    uncertainty_results = uncertainty_estimation(best_model, val_loader)
    print(f"\nUncertainty Estimation:")
    print(f"Mean prediction entropy: {uncertainty_results['mean_entropy']:.4f}")
    print(f"Accuracy with MC Dropout: {uncertainty_results['accuracy']:.4f}")
    
    # Evaluate robustness to different noise levels
    snr_results = simulate_communication_constraints(best_model, val_loader)
    
    # Plot SNR vs accuracy
    plot_snr_vs_accuracy(snr_results)
    
    print("\nFinal Results Summary:")
    print(f"Model Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {cv_avg:.4f} ± {cv_std:.4f}")
    print(f"Radar Importance: 24GHz={results['radar_weights'][0]:.3f}, "
          f"77GHz={results['radar_weights'][1]:.3f}, "
          f"Xethru={results['radar_weights'][2]:.3f}")
    print(f"Robustness at 0dB SNR: {snr_results[0]:.4f}")


if __name__ == '__main__':
    main()