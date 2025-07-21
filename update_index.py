import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import copy
from thop import profile
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE

# Constants
NUM_CLASSES = 11
FEATURE_DIM = 256  # Reduced from 512

class RadarSpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None, radars=['24GHz', '77GHz', 'Xethru'], apply_augmentation=False):
        self.root_dir = root_dir
        self.transform = transform
        self.radars = radars
        self.apply_augmentation = apply_augmentation
        
        self.activities = sorted([d for d in os.listdir(os.path.join(root_dir, radars[0])) 
                               if os.path.isdir(os.path.join(root_dir, radars[0], d))])
        
        self.samples = []
        # Build dataset with samples from all radars
        for activity_idx, activity in enumerate(self.activities):
            sample_files = sorted(os.listdir(os.path.join(root_dir, radars[0], activity)))
            for sample in sample_files:
                # Ensure all radars have this sample
                valid = True
                for radar in radars[1:]:
                    if not os.path.exists(os.path.join(root_dir, radar, activity, sample)):
                        valid = False
                        break
                
                if valid:
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
            t = int(spectrogram.shape[2] * 0.1)  # Mask up to 10% of time steps
            t0 = torch.randint(0, spectrogram.shape[2] - t, (1,)).item()
            spectrogram[:, :, t0:t0+t] = 0
        
        if torch.rand(1).item() < 0.5:
            # Apply frequency masking
            f = int(spectrogram.shape[1] * 0.1)  # Mask up to 10% of frequency bins
            f0 = torch.randint(0, spectrogram.shape[1] - f, (1,)).item()
            spectrogram[:, f0:f0+f, :] = 0
            
        # Small random noise addition for robustness
        if torch.rand(1).item() < 0.3:
            noise = torch.randn_like(spectrogram) * 0.02
            spectrogram = spectrogram + noise
            spectrogram = torch.clamp(spectrogram, 0, 1)
            
        return spectrogram

# Custom lightweight backbones for frequency-specific feature extraction
class LightweightFrequencyEncoder(nn.Module):
    def __init__(self, input_channels=3, frequency_type='low'):
        """
        Custom CNN designed for specific radar frequency ranges
        Args:
            input_channels: Number of input channels (3 for RGB spectrograms)
            frequency_type: 'low', 'high', or 'ultra' for different frequency characteristics
        """
        super(LightweightFrequencyEncoder, self).__init__()
        
        # Common initial layers
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Frequency-specific layers
        if frequency_type == 'low':  # 24GHz - deeper for temporal patterns
            self.frequency_layers = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, FEATURE_DIM, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(FEATURE_DIM),
                nn.ReLU(inplace=True),
            )
        elif frequency_type == 'high':  # 77GHz - optimized for spatial resolution
            self.frequency_layers = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # Larger kernel for better spatial resolution
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, FEATURE_DIM, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(FEATURE_DIM),
                nn.ReLU(inplace=True),
            )
        else:  # Xethru - optimized for penetration/motion detection
            self.frequency_layers = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, FEATURE_DIM, kernel_size=1, stride=1, padding=0),  # 1x1 conv to capture motion patterns
                nn.BatchNorm2d(FEATURE_DIM),
                nn.ReLU(inplace=True),
            )
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Apply dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.frequency_layers(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        return x.view(-1, FEATURE_DIM)

class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, num_heads=4):
        super(MultiHeadAttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # One attention projection per radar type
        self.attention_24g = nn.Linear(feature_dim, num_heads)
        self.attention_77g = nn.Linear(feature_dim, num_heads)
        self.attention_xethru = nn.Linear(feature_dim, num_heads)
        
        self.min_contribution = 0.1
        
    def forward(self, f24, f77, fx):
        batch_size = f24.size(0)
        
        # Check if batch sizes match - this should be added
        if f24.size(0) != f77.size(0) or f24.size(0) != fx.size(0):
            # Find the minimum batch size
            min_batch = min(f24.size(0), f77.size(0), fx.size(0))
            # Truncate to match the smallest batch
            f24 = f24[:min_batch]
            f77 = f77[:min_batch]
            fx = fx[:min_batch]
        
        # Calculate attention scores for each radar
        a24 = self.attention_24g(f24)  # [batch_size, num_heads]
        a77 = self.attention_77g(f77)  # [batch_size, num_heads]
        ax = self.attention_xethru(fx)  # [batch_size, num_heads]
        
        # Rest of the function remains the same
        head_size = self.feature_dim // self.num_heads
        fused = torch.zeros_like(f24)
        all_weights = []
        
        # Process each head separately
        for h in range(self.num_heads):
            # Get weights for this head
            head_weights = torch.stack([
                a24[:, h],
                a77[:, h],
                ax[:, h]
            ], dim=1)  # [batch_size, 3]
            
            # Softmax to normalize
            head_weights = F.softmax(head_weights, dim=1)
            
            # Apply minimum constraint
            head_weights = torch.clamp(head_weights, min=self.min_contribution)
            head_weights = head_weights / head_weights.sum(dim=1, keepdim=True)
            
            # Store weights
            all_weights.append(head_weights)
            
            # Start and end indices for this head's features
            start_idx = h * head_size
            end_idx = (h + 1) * head_size
            
            # Apply weights to features
            head_fused = (
                head_weights[:, 0:1] * f24[:, start_idx:end_idx] +
                head_weights[:, 1:2] * f77[:, start_idx:end_idx] +
                head_weights[:, 2:3] * fx[:, start_idx:end_idx]
            )
            
            # Add to output
            fused[:, start_idx:end_idx] = head_fused
            
        return fused, all_weights

class ActivityClassifier(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, num_classes=NUM_CLASSES, dropout_rate=0.5):
        super(ActivityClassifier, self).__init__()
        
        # Add more regularization and complexity
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        # Monte Carlo dropout for uncertainty estimation
        self.enable_mc_dropout = False
        self.mc_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        if self.enable_mc_dropout and self.training:
            x = self.mc_dropout(x)
        return self.classifier(x)
    
    def enable_uncertainty(self, enable=True):
        """Enable Monte Carlo dropout for uncertainty estimation"""
        self.enable_mc_dropout = enable

class ImprovedMultiRadarFusion(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feature_dim=FEATURE_DIM):
        super(ImprovedMultiRadarFusion, self).__init__()
        
        # Use custom frequency-specific encoders instead of ResNet18
        self.radar_24g_encoder = LightweightFrequencyEncoder(frequency_type='low')
        self.radar_77g_encoder = LightweightFrequencyEncoder(frequency_type='high')
        self.xethru_encoder = LightweightFrequencyEncoder(frequency_type='ultra')
        
        # Multi-head attention fusion
        self.attention_fusion = MultiHeadAttentionFusion(feature_dim=feature_dim)
        
        # Activity classifier
        self.classifier = ActivityClassifier(feature_dim=feature_dim, num_classes=num_classes)
        
        # L2 regularization
        self.l2_reg_factor = 1e-4
        
    def forward(self, inputs):
        r24, r77, xethru = inputs
        
        # Extract features from each radar
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Fusion with multi-head attention
        fused, _ = self.attention_fusion(f24, f77, fx)
        
        # Classification
        outputs = self.classifier(fused)
        return outputs
    
    def get_attention_weights(self, inputs):
        r24, r77, xethru = inputs
        
        # Extract features from each radar
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Get attention weights from fusion
        _, all_head_weights = self.attention_fusion(f24, f77, fx)
        
        # Average weights across heads and batches
        avg_weights = {
            '24GHz': torch.mean(torch.stack([w[:, 0] for w in all_head_weights]), dim=0),
            '77GHz': torch.mean(torch.stack([w[:, 1] for w in all_head_weights]), dim=0),
            'Xethru': torch.mean(torch.stack([w[:, 2] for w in all_head_weights]), dim=0)
        }
        
        return avg_weights
    
    def l2_regularization_loss(self):
        """Calculate L2 regularization loss for all parameters"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg_factor * l2_loss

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model([r24, r77, xethru])
            loss = criterion(outputs, labels)
            
            # Add L2 regularization
            l2_loss = model.l2_regularization_loss()
            loss += l2_loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                r24, r77, xethru = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                outputs = model([r24, r77, xethru])
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Early stopping with patience
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            # Change the save path to current directory
            torch.save(model.state_dict(), './best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return history

def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []  # Store softmax probabilities for uncertainty estimation
    attention_weights = {'24GHz': [], '77GHz': [], 'Xethru': []}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Get model outputs
            outputs = model([r24, r77, xethru])
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Get attention weights
            batch_weights = model.get_attention_weights([r24, r77, xethru])
            for k, v in batch_weights.items():
                attention_weights[k].extend(v.cpu().numpy())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
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
    
    # Aggregate attention weights
    for k in attention_weights:
        attention_weights[k] = np.array(attention_weights[k])
    
    # Attention weights per class
    class_attention = {radar: [] for radar in attention_weights}
    for class_idx in range(len(class_names)):
        class_mask = np.array(all_labels) == class_idx
        for radar in attention_weights:
            if np.any(class_mask):
                class_attention[radar].append(np.mean(attention_weights[radar][class_mask]))
            else:
                class_attention[radar].append(0)
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'certainty': certainty,
        'attention_weights': attention_weights,
        'class_attention': class_attention
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
            all_labels.extend(labels.cpu().numpy())
            
            # Collect multiple predictions with dropout enabled
            batch_samples = []
            for _ in range(num_samples):
                outputs = model([r24, r77, xethru])
                probs = F.softmax(outputs, dim=1)
                batch_samples.append(probs.cpu().numpy())
            
            # Average predictions
            mean_probs = np.mean(np.array(batch_samples), axis=0)
            sample_predictions.append(np.array(batch_samples))
            
            # Get final predictions
            preds = np.argmax(mean_probs, axis=1)
            all_predictions.extend(preds)
    
    # Concatenate all sample predictions
    all_samples = np.vstack([p.reshape(-1, p.shape[-1]) for p in sample_predictions])
    
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
        'mean_entropy': np.mean(entropy),
        'mean_probs': mean_probs
    }

def quantize_model(model, data_loader=None):
    """Quantize model to reduce size and improve inference speed"""
    # Make a copy of the model for quantization
    model.eval()
    
    # Create quantization configuration
    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_to_quantize = copy.deepcopy(model).cpu()
    
    # Prepare model for quantization
    torch.quantization.prepare(model_to_quantize, inplace=True)
    
    # Calibrate model if data_loader is provided (recommended)
    if data_loader:
        with torch.no_grad():
            for inputs, _ in data_loader:
                r24, r77, xethru = [x.cpu() for x in inputs]
                model_to_quantize([r24, r77, xethru])
    
    # Convert model to quantized version
    torch.quantization.convert(model_to_quantize, inplace=True)
    
    return model_to_quantize

def compare_model_sizes(original_model, quantized_model):
    """Compare sizes of original and quantized models"""
    import os
    import tempfile
    
    # Create temporary files
    with tempfile.NamedTemporaryFile() as orig_file, tempfile.NamedTemporaryFile() as quant_file:
        # Save the original model
        torch.save(original_model.state_dict(), orig_file.name)
        original_size = os.path.getsize(orig_file.name) / (1024 * 1024)
        
        # Save the quantized model
        torch.save(quantized_model.state_dict(), quant_file.name)
        quantized_size = os.path.getsize(quant_file.name) / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")
    
    return {
        'original_size_MB': original_size, 
        'quantized_size_MB': quantized_size,
        'reduction_percentage': (1 - quantized_size/original_size)*100
    }

def simulate_communication_constraints(model, val_loader, snr_levels=[30, 20, 10, 5, 0, -5, -10]):
    """Simulate performance under different SNR conditions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for snr in snr_levels:
        # Function to add noise based on SNR
        def add_noise(signal, snr_db):
            # Calculate signal power
            signal_power = torch.mean(signal**2)
            # Avoid division by zero
            if signal_power < 1e-10:
                return signal
            # Calculate noise power from SNR
            noise_power = signal_power / (10**(snr_db/10))
            # Generate and add noise
            noise = torch.randn_like(signal) * torch.sqrt(noise_power)
            return signal + noise
        
        # Evaluate with noise
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                # Move to device
                clean_inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # Add noise to simulate communication channel
                noisy_inputs = [add_noise(x, snr) for x in clean_inputs]
                
                # Forward pass
                outputs = model(noisy_inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        if total > 0:
            results[snr] = correct / total
    
    return results

def measure_energy_consumption(model, sample_inputs, num_iterations=100):
    """Estimate energy consumption for model inference (simulation-based)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Move inputs to device
    inputs = [x.to(device) for x in sample_inputs]
    
    # Warm up
    for _ in range(10):
        _ = model(inputs)
    
    # Measure inference time
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(inputs)
    end_time = time.time()
    
    # Calculate inference time
    avg_inference_time = (end_time - start_time) / num_iterations
    
    # Estimate energy based on device
    if device.type == 'cuda':
        # Rough estimate for GPU energy consumption
        # Assume 150W power draw for an average GPU
        estimated_energy = 150 * avg_inference_time  # Energy in Joules (W×s)
    else:
        # Rough estimate for CPU energy consumption
        # Assume 15W power draw for an average CPU core
        estimated_energy = 15 * avg_inference_time  # Energy in Joules
    
    return {
        'inference_time_ms': avg_inference_time * 1000,
        'estimated_energy_J': estimated_energy,
        'estimated_energy_per_inference_mJ': estimated_energy * 1000
    }

def gradient_class_activation_map(model, inputs, target_class=None):
    """Generate Grad-CAM visualization to highlight important regions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Move inputs to device
    r24, r77, xethru = [x.unsqueeze(0).to(device) for x in inputs]
    
    # Enable gradient calculation for feature maps
    r24.requires_grad = True
    r77.requires_grad = True
    xethru.requires_grad = True
    
    # Forward pass
    outputs = model([r24, r77, xethru])
    
    # If target class is not specified, use the predicted class
    if target_class is None:
        target_class = outputs.argmax(dim=1).item()
    
    # Zero gradients
    model.zero_grad()
    
    # Target for backprop
    one_hot = torch.zeros_like(outputs)
    one_hot[0, target_class] = 1
    
    # Backward pass
    outputs.backward(gradient=one_hot, retain_graph=True)
    
    # Get gradients and feature maps
    grad_24 = r24.grad.detach()
    grad_77 = r77.grad.detach()
    grad_xethru = xethru.grad.detach()
    
    # Get activation maps (last convolutional layer outputs)
    # For simplicity, we'll use the gradient magnitude as our attention map
    activation_24 = torch.mean(torch.abs(grad_24), dim=1)
    activation_77 = torch.mean(torch.abs(grad_77), dim=1)
    activation_xethru = torch.mean(torch.abs(grad_xethru), dim=1)
    
    # Normalize activations
    activation_24 = F.interpolate(activation_24.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    activation_77 = F.interpolate(activation_77.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    activation_xethru = F.interpolate(activation_xethru.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    
    # Normalize between 0 and 1
    activation_24 = (activation_24 - activation_24.min()) / (activation_24.max() - activation_24.min() + 1e-8)
    activation_77 = (activation_77 - activation_77.min()) / (activation_77.max() - activation_77.min() + 1e-8)
    activation_xethru = (activation_xethru - activation_xethru.min()) / (activation_xethru.max() - activation_xethru.min() + 1e-8)
    
    return {
        '24GHz': activation_24.squeeze().cpu().numpy(),
        '77GHz': activation_77.squeeze().cpu().numpy(),
        'Xethru': activation_xethru.squeeze().cpu().numpy()
    }

def visualize_gradcam(model, dataset, class_names, sample_idx=0):
    """Visualize Grad-CAM for sample inputs"""
    # Get a sample
    inputs, label = dataset[sample_idx]
    class_name = class_names[label]
    
    # Generate Grad-CAM
    cam_maps = gradient_class_activation_map(model, inputs, label)
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    radar_types = ['24GHz', '77GHz', 'Xethru']
    
    for i, radar in enumerate(radar_types):
        # Original image
        plt.subplot(2, 3, i+1)
        img = inputs[i].permute(1, 2, 0).cpu().numpy()
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Original {radar}")
        plt.axis('off')
        
        # Grad-CAM overlay
        plt.subplot(2, 3, i+4)
        plt.imshow(img)
        plt.imshow(cam_maps[radar], alpha=0.5, cmap='jet')
        plt.title(f"Grad-CAM {radar}")
        plt.axis('off')
    
    plt.suptitle(f"Class: {class_name}")
    plt.tight_layout()
    plt.savefig(f'gradcam_{class_name}.png')
    plt.show()

def perform_cross_validation(model_class, dataset, k=5, num_epochs=10):
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
        criterion = nn.CrossEntropyLoss()
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
                outputs = model(radar_inputs)
                loss = criterion(outputs, labels) + model.l2_regularization_loss()
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
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
                    
                    # Forward pass
                    outputs = model(radar_inputs)
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
                outputs = model(radar_inputs)
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

def ablation_study(model, val_loader, class_names):
    """Test performance with different combinations of radars"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Define radar combinations to test
    combinations = [
        {'name': 'All Radars', 'mask': [1, 1, 1]},
        {'name': '24GHz Only', 'mask': [1, 0, 0]},
        {'name': '77GHz Only', 'mask': [0, 1, 0]},
        {'name': 'Xethru Only', 'mask': [0, 0, 1]},
        {'name': '24GHz + 77GHz', 'mask': [1, 1, 0]},
        {'name': '24GHz + Xethru', 'mask': [1, 0, 1]},
        {'name': '77GHz + Xethru', 'mask': [0, 1, 1]}
    ]
    
    results = {}
    confusion_matrices = {}
    
    for combo in combinations:
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                # Move to device
                radar_inputs = [inp.to(device) for inp in batch_inputs]
                batch_labels = batch_labels.to(device)
                
                # Generate masked features
                for i, use_radar in enumerate(combo['mask']):
                    if not use_radar:
                        radar_inputs[i] = torch.zeros_like(radar_inputs[i])
                
                # Forward pass
                outputs = model(radar_inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        results[combo['name']] = accuracy
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        confusion_matrices[combo['name']] = cm
    
    # Plot results
    plt.figure(figsize=(12, 6))
    names = list(results.keys())
    accuracies = list(results.values())
    
    bars = plt.bar(names, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Radar Combination')
    plt.xticks(rotation=45)
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig('radar_combinations_accuracy.png')
    plt.show()
    
    return results, confusion_matrices

def visualize_features(model, val_loader, class_names):
    """Visualize feature space to understand separability using t-SNE"""
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
            fused, _ = model.attention_fusion(f24, f77, fx)
            
            features.append(fused.cpu().numpy())
            labels_list.append(labels.numpy())
    
    if features:
        features = np.vstack(features)
        labels_array = np.concatenate(labels_list)
        
        # Apply t-SNE
        print("Applying t-SNE dimensionality reduction...")
        tsne = TSNE(n_components=2, random_state=42, verbose=1)
        features_tsne = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 10))
        for i, activity in enumerate(class_names):
            mask = labels_array == i
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=activity)
        
        plt.legend()
        plt.title('t-SNE Visualization of Fused Features')
        plt.savefig('tsne_features.png')
        plt.show()
    else:
        print("No features extracted. Check if the validation loader is empty.")

def plot_attention_heatmap(class_attention, class_names):
    """Plot radar attention heatmap by activity class"""
    data = np.array([
        class_attention['24GHz'], 
        class_attention['77GHz'],
        class_attention['Xethru']
    ])
    
    plt.figure(figsize=(14, 6))
    plt.imshow(data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Attention Weight')
    
    # Add text annotations
    for i in range(len(data)):
        for j in range(len(data[0])):
            plt.text(j, i, f"{data[i][j]:.3f}", 
                     ha="center", va="center", 
                     color="white" if data[i][j] > 0.5 else "black")
    
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(3), ['24GHz', '77GHz', 'Xethru'])
    plt.title('Radar Importance by Activity Class')
    plt.tight_layout()
    plt.savefig('radar_activity_heatmap.png')
    plt.show()

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

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
    
    # Create dataset with augmentation for training
    train_dataset = RadarSpectrogramDataset(
        root_dir=r"E:\SpectrogramFusion\11_class_activity_data", 
        transform=transform,
        apply_augmentation=True
    )
    
    # Create dataset without augmentation for validation
    val_dataset = RadarSpectrogramDataset(
        root_dir=r"E:\SpectrogramFusion\11_class_activity_data", 
        transform=transform,
        apply_augmentation=False
    )
    
    # Get activity class names
    class_names = train_dataset.activities
    
    # Split into train/validation sets (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(val_dataset) - train_size
    train_subset, _ = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    _, val_subset = torch.utils.data.random_split(
        val_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders with drop_last=True
    train_loader = DataLoader(
        train_subset, batch_size=16, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=16, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    # Initialize the improved model
    model = ImprovedMultiRadarFusion(num_classes=len(class_names))
    
    # Print model architecture summary
    print(f"Model architecture:")
    print(f"- Radar encoders: Custom lightweight frequency-specific encoders")
    print(f"- Fusion mechanism: Multi-head attention with minimum contribution constraint")
    print(f"- Classifier: Multi-layer with regularization and uncertainty estimation")
    print(f"- Feature dimension: {FEATURE_DIM}")
    
    # Train the model
    print("Training model...")
    history = train_model(model, train_loader, val_loader, num_epochs=50)
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    best_model = ImprovedMultiRadarFusion(num_classes=len(class_names))
    best_model.load_state_dict(torch.load('./best_model.pth'))
    
    # Evaluate the model
    print("Evaluating model...")
    results = evaluate_model(best_model, val_loader, class_names)
    
    # Print evaluation results
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    
    print("\nClassification Report:")
    for class_name, metrics in results['classification_report'].items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{class_name}: {metrics}")

    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    for i, class_name in enumerate(class_names):
          print(f"{class_name}: Precision={results['precision'][i]:.4f}, "
                f"Recall={results['recall'][i]:.4f}, F1={results['f1'][i]:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(results['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = results['confusion_matrix'].max() / 2.
    for i in range(results['confusion_matrix'].shape[0]):
        for j in range(results['confusion_matrix'].shape[1]):
            plt.text(j, i, format(results['confusion_matrix'][i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if results['confusion_matrix'][i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Plot radar importance by activity class
    plot_attention_heatmap(results['class_attention'], class_names)
    
    # Print overall radar importance
    print("\nOverall Radar Importance:")
    for radar in ['24GHz', '77GHz', 'Xethru']:
        importance = np.mean(results['class_attention'][radar])
        print(f"{radar}: {importance:.4f}")
    
    # Perform uncertainty estimation
    uncertainty_results = uncertainty_estimation(best_model, val_loader)
    print(f"\nUncertainty Estimation:")
    print(f"Mean prediction entropy: {uncertainty_results['mean_entropy']:.4f}")
    print(f"Accuracy with MC Dropout: {uncertainty_results['accuracy']:.4f}")
    
    # Measure resource usage
    sample_inputs = [torch.randn(1, 3, 224, 224) for _ in range(3)]
    energy_metrics = measure_energy_consumption(best_model, sample_inputs)
    print(f"\nEnergy Consumption Metrics:")
    print(f"Average inference time: {energy_metrics['inference_time_ms']:.2f} ms")
    print(f"Estimated energy per inference: {energy_metrics['estimated_energy_per_inference_mJ']:.2f} mJ")
    
    # REMOVE quantization and size comparison
    # quantized_model = quantize_model(best_model.cpu(), val_loader)
    # size_comparison = compare_model_sizes(best_model.cpu(), quantized_model)
    
    # Perform ablation study
    ablation_results, _ = ablation_study(best_model, val_loader, class_names)
    
    # Simulate communication constraints
    communication_results = simulate_communication_constraints(best_model, val_loader)
    
    # Plot communication simulation results
    plt.figure(figsize=(10, 6))
    snrs = list(communication_results.keys())
    accuracies = [communication_results[s] for s in snrs]
    
    plt.plot(snrs, accuracies, marker='o', linestyle='-')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('Performance under Communication Constraints')
    plt.grid(True)
    
    # Add data labels
    for x, y in zip(snrs, accuracies):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", 
                     xytext=(0,10), ha='center')
    
    plt.savefig('communication_simulation.png')
    plt.show()
    
    # Visualize feature space
    visualize_features(best_model, val_loader, class_names)
    
    # Generate Grad-CAM visualizations for sample data points
    for idx in range(5):  # Visualize 5 samples
        visualize_gradcam(best_model, val_subset, class_names, sample_idx=idx)
        
    # Cross-validation
    cv_results, cv_avg, cv_std = perform_cross_validation(
        ImprovedMultiRadarFusion, val_dataset, k=5, num_epochs=10
    )
    
    print("\nFinal Results Summary:")
    print(f"Model Accuracy: {results['accuracy']:.4f}")
    print(f"Cross-validation Accuracy: {cv_avg:.4f} ± {cv_std:.4f}")
    # REMOVE model size reduction line
    # print(f"Model Size Reduction: {size_comparison['reduction_percentage']:.2f}%")
    print(f"Radar Importance Balance: 24GHz={np.mean(results['class_attention']['24GHz']):.3f}, "
          f"77GHz={np.mean(results['class_attention']['77GHz']):.3f}, "
          f"Xethru={np.mean(results['class_attention']['Xethru']):.3f}")
    print(f"Communication Robustness at 0dB SNR: {communication_results[0]:.4f}")

if __name__ == '__main__':
    main()