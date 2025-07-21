import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.multiprocessing as mp



#This is a standalone class that handles the loading and organization of radar spectrogram data.
class RadarSpectrogramDataset(Dataset):
    """
    This class handles loading and organizing radar spectrogram data.
    It applies transformations to the images and provides samples for training and evaluation.
    """
    def __init__(self, root_dir, transform=None, radars=['24GHz', '77GHz', 'Xethru']):
        """
        Initialize the dataset.
        - root_dir: Directory containing radar data.
        - transform: Image transformations (e.g., resizing, flipping).
        - radars: List of radar types (e.g., 24GHz, 77GHz, Xethru).
        """
        self.root_dir = root_dir
        self.transform = transform 
        self.radars = radars 
        self.activities = sorted([d for d in os.listdir(os.path.join(root_dir, radars[0])) 
                               if os.path.isdir(os.path.join(root_dir, radars[0], d))])
        
        
        self.samples = []
        for activity_idx, activity in enumerate(self.activities):
            sample_files = sorted(os.listdir(os.path.join(root_dir, radars[0], activity)))
            for sample in sample_files:
                valid = True
                for radar in radars[1:]:
                    if not os.path.exists(os.path.join(root_dir, radar, activity, sample)):
                        valid = False
                        break
                
                if valid:
                    self.samples.append((activity, activity_idx, sample))
    
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retrieve a sample (activity, label, and file name) for a given index.
        - idx: Index of the sample.
        - Returns: List of spectrograms and the activity label.
        """
        activity, label, sample = self.samples[idx]
        
        
        radar_spectrograms = []
        for radar in self.radars:
            img_path = os.path.join(self.root_dir, radar, activity, sample)
            image = Image.open(img_path).convert('RGB')  
            if self.transform:
                image = self.transform(image)
            
            radar_spectrograms.append(image)
        return radar_spectrograms, label

# Define separate transformations for training and validation
transform_train = transforms.Compose([
    transforms.Resize((224, 224)), # Resize all images to 224x224 pixels
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flip images horizontally with 50% probability
    # Apply small random rotation (max ±5 degrees)
    # Translate image randomly by up to 5% in x and y directions
    # Scale image randomly between 95% and 105% of original size
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
    # Adjust brightness randomly within ±10%
    # Adjust contrast randomly within ±10%
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
      # Convert image to PyTorch tensor (scales pixel values to [0,1])
    transforms.ToTensor(),
    # Normalize using ImageNet mean values (for RGB channels)
    # Normalize using ImageNet standard deviation values
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    # Define transformations for validation dataset (no augmentation, only normalization)
    # Resize all images to 224x224 pixels
    transforms.Resize((224, 224)),
     # Convert image to PyTorch tensor
    transforms.ToTensor(),
      # Normalize using ImageNet statistics
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Create full datasets with different transformations for training and validation
# Here we're initializing two dataset instances with the same data but different transformations
# train_full_dataset uses data augmentation for better generalization
# val_full_dataset uses only normalization for consistent evaluation
train_full_dataset = RadarSpectrogramDataset(root_dir=r"C:\Users\mk305r\Desktop\Multi-StreamRadar\11_class_activity_data", transform=transform_train)
val_full_dataset = RadarSpectrogramDataset(root_dir=r"C:\Users\mk305r\Desktop\Multi-StreamRadar\11_class_activity_data", transform=transform_val)

# Create a third dataset instance that will be used for creating train/validation splits
# This allows more flexibility for cross-validation and other evaluation techniques
dataset = RadarSpectrogramDataset(root_dir=r"C:\Users\mk305r\Desktop\Multi-StreamRadar\11_class_activity_data", transform=transform_val)


# Implement stratified split to ensure class balance in both training and validation sets
from sklearn.model_selection import train_test_split

# Create a list of all indices in the dataset
indices = list(range(len(dataset)))
# Extract the class labels for each sample to enable stratified splitting
labels = [dataset.samples[i][1] for i in indices]
# Split indices into training (80%) and validation (20%) sets while preserving class distribution
# random_state=42 ensures reproducibility of the split
train_indices, val_indices = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

# Create Subset datasets using the split indices
# This creates views of the original dataset rather than copying the data
# Each subset will apply the appropriate transformations when accessed
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

# Create data loaders that will handle batching and shuffling during training/validation
# batch_size=16: Process 16 samples at once through the model
# shuffle=True: Randomize sample order each epoch (for training only)
# num_workers=4: Use 4 parallel processes to load and preprocess data
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RadarEncoder(nn.Module):
    """
    This class encodes radar spectrograms into feature vectors using a pre-trained ResNet18 model.
    """
    # This class encodes radar spectrograms into feature vectors using a pre-trained ResNet18 model.
    # It serves as a feature extractor, transforming radar images into meaningful embeddings.
    def __init__(self, pretrained=True):
        """
        Initialize the encoder.
        - pretrained: If True, use pre-trained weights from ImageNet.
        """
        # pretrained parameter determines whether to use ImageNet weights
        super(RadarEncoder, self).__init__()
        # Use ResNet18 as backbone, removing the final FC layer
        # Update for newer PyTorch versions:
        if pretrained:
            # Use pre-trained weights from ImageNet for transfer learning
            # This leverages knowledge from natural images to help with radar spectrograms
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            # Start with random initialization if pretrained=False
            weights = None
         # Load the ResNet18 model with appropriate weights
        resnet = models.resnet18(weights=weights)
        # Remove the final fully connected layer (classification layer) from ResNet
        # This gives us the feature extraction portion without the classification head
        # The [:-1] slicing removes the last layer from the children modules
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512 # Feature dimension (output size)
        
    def forward(self, x):
        """
        Forward pass: Convert input radar spectrograms into feature vectors.
        - x: Input radar spectrogram.
        - Returns: Feature vector.
        """
        # Forward pass: transform input radar spectrograms into feature vectors
        # x shape: [batch_size, channels, height, width]
        
        # Pass the input through the encoder network
        features = self.encoder(x)
        # features shape: [batch_size, 512, 1, 1] (after global average pooling in ResNet)
        
        # Reshape the output to flatten the spatial dimensions
        # Returns tensor of shape [batch_size, 512]
        return features.view(-1, self.feature_dim)

class AttentionFusion(nn.Module):
    """
    This class fuses features from multiple radars using attention mechanisms.
    """
    # This class fuses features from multiple radars using attention mechanisms.
    # It dynamically weights the importance of each radar signal based on their feature content.

    def __init__(self, feature_dim=512):
        """
        Initialize the attention fusion module.
        - feature_dim: Dimension of input feature vectors.
        """
        # Initialize the parent class (nn.Module)
        super(AttentionFusion, self).__init__()
        # Store the dimension of input feature vectors
        self.feature_dim = feature_dim
        
        # Create learnable attention layers for each radar type
        # Each layer maps a feature vector to a single attention score
        # Linear layer inputs: [batch_size, feature_dim]
        # Linear layer outputs: [batch_size, 1] (a scalar attention weight)
        self.attention_24g = nn.Linear(feature_dim, 1) 
        self.attention_77g = nn.Linear(feature_dim, 1)
        self.attention_xethru = nn.Linear(feature_dim, 1)
        
    def forward(self, f24, f77, fx):
        """
        Forward pass: Combine features from three radars using attention weights.
        - f24, f77, fx: Feature vectors from 24GHz, 77GHz, and Xethru radars.
        - Returns: Fused feature vector.
        """
        # Forward pass: compute attention-weighted combination of radar features
        # Input shapes: each is [batch_size, feature_dim]
        
        # Calculate attention scores for each radar modality
        # Sigmoid ensures values are between 0 and 1 (representing importance)
        a24 = torch.sigmoid(self.attention_24g(f24))
        a77 = torch.sigmoid(self.attention_77g(f77))
        ax = torch.sigmoid(self.attention_xethru(fx))
        
        # Normalize attention weights so they sum to 1 (softmax-like normalization)
        # This ensures proper weighting across all radar types
        attn_sum = a24 + a77 + ax
        a24 = a24 / attn_sum
        a77 = a77 / attn_sum
        ax = ax / attn_sum
        
        # Weighted fusion: combine features using normalized attention weights
        # This is an element-wise weighted sum, with each radar contributing
        # according to its computed importance for the current sample
        fused = a24 * f24 + a77 * f77 + ax * fx
        # Return the fused feature vector, shape: [batch_size, feature_dim]
        return fused

class ActivityClassifier(nn.Module):
    """
    This class classifies the fused features into activity labels.
    """
# This class classifies the fused features into activity labels.
# It implements a multi-layer perceptron with regularization techniques
# to prevent overfitting and improve generalization.

    def __init__(self, feature_dim=512, num_classes=11):
        """
        Initialize the classifier.
        - feature_dim: Dimension of input feature vector.
        - num_classes: Number of output classes (activities).
        """
        super(ActivityClassifier, self).__init__()
        # Define fully connected layers for classification
       # This creates a narrowing architecture (512 -> 256 -> 128 -> 11)
       # which compresses information and extracts increasingly abstract representations
        
        # First fully connected layer
        # Input: feature_dim (512), Output: 256
        self.fc1 = nn.Linear(feature_dim, 256)

        # Batch normalization after first layer
        # Stabilizes learning by normalizing activations
        # Reduces internal covariate shift
        self.bn1 = nn.BatchNorm1d(256)

        # Aggressive dropout (70%) after first layer
        # Randomly zeroes 70% of inputs during training
        # Forces the network to learn redundant representations
        self.dropout1 = nn.Dropout(0.7)  

        # Second fully connected layer
        # Further compresses features
        self.fc2 = nn.Linear(256, 128) 

        # Batch normalization after second layer
        # Maintains normalized activations throughout network
        self.bn2 = nn.BatchNorm1d(128)   

        # Moderate dropout (50%) after second layer
        # Less aggressive than first dropout to maintain
        # sufficient information flow to output
        self.dropout2 = nn.Dropout(0.5)  

        # Final classification layer
         # Maps to output classes (default: 11 activities)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass: Predict the activity class.
        - x: Fused feature vector.
        - Return
        """
        # Forward pass through the classifier
        # Input x shape: [batch_size, feature_dim]
       
        # First layer with ReLU activation, batch norm, and dropout
        x = F.relu(self.bn1(self.fc1(x))) # Apply layer, then batch norm, then ReLU
        x = self.dropout1(x) ## Apply dropout (only active during training)
        # Second layer with ReLU activation, batch norm, and dropout
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        # Final classification layer (no activation - raw logits)
        # Softmax will be applied in the loss function for numerical stability
        x = self.fc3(x)  # Output shape: [batch_size, num_classes]
        return x # Return class logits (unnormalized predictions)

class MultiRadarFusion(nn.Module):
    """
    This class combines the radar encoders, attention fusion, and classifier into one model.
    """

    # This class combines the radar encoders, attention fusion, and classifier into one model.
    # It implements an end-to-end architecture for multi-modal radar activity recognition.

    def __init__(self, num_classes=11, pretrained=True):
        """
        Initialize the model.
        - num_classes: Number of output classes (activities).
        - pretrained: If True, use pre-trained weights for the encoders.
        """
        # Initialize parent class
        super(MultiRadarFusion, self).__init__()

        # Define encoders for each radar modality
        # Each encoder is an instance of RadarEncoder that processes one radar type
        # All encoders share the same architecture (ResNet18) but have separate weights
        # pretrained=True means they start with ImageNet weights for transfer learning
        self.radar_24g_encoder = RadarEncoder(pretrained=pretrained)
        self.radar_77g_encoder = RadarEncoder(pretrained=pretrained)
        self.xethru_encoder = RadarEncoder(pretrained=pretrained)

        # Get the feature dimension from the encoder (512 for ResNet18)
        feature_dim = self.radar_24g_encoder.feature_dim
        # Create the attention fusion module to combine features from different radars
        # This dynamically weights each radar's contribution based on content
        self.attention_fusion = AttentionFusion(feature_dim=feature_dim)
         # Create the activity classifier that maps fused features to activity classes
        self.classifier = ActivityClassifier(feature_dim=feature_dim, num_classes=num_classes)

        
    def forward(self, inputs):
        """
        Forward pass: Process input radar spectrograms and predict the activity class.
        - inputs: List of radar spectrograms (24GHz, 77GHz, Xethru).
        - Returns: Predicted activity class.
        """
        # Forward pass through the entire network
        # inputs is a list of three radar spectrograms [24GHz, 77GHz, Xethru]
        
        # Unpack inputs (spectrograms from each radar type)
        r24, r77, xethru = inputs
        
        # Extract features from each radar modality separately
        # Each encoder transforms a spectrogram into a 512-dim feature vector
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Fusion with attention - combine features with learned weights
        # This dynamically determines which radar provides the most useful information
        fused = self.attention_fusion(f24, f77, fx)
        
        # Classification - map fused features to activity classes
        # Shape: [batch_size, num_classes]
        outputs = self.classifier(fused)
        # Return class logits
        return outputs
    
    def get_attention_weights(self, inputs):
        """
        Extract attention weights for analysis/visualization.
        - inputs: List of radar spectrograms.
        - Returns: Dictionary of attention weights for each radar.
        """
        # This method extracts the attention weights for analysis/visualization
        # It shows how much each radar contributes to the decision for a given input
        
        # Unpack inputs
        r24, r77, xethru = inputs
        
        # Extract features from each radar (same as in forward pass)
        f24 = self.radar_24g_encoder(r24)
        f77 = self.radar_77g_encoder(r77)
        fx = self.xethru_encoder(xethru)
        
        # Calculate attention weights using the same mechanism as in AttentionFusion
        # but exposed for external analysis

        a24 = torch.sigmoid(self.attention_fusion.attention_24g(f24))
        a77 = torch.sigmoid(self.attention_fusion.attention_77g(f77))
        ax = torch.sigmoid(self.attention_fusion.attention_xethru(fx))
        
        # Normalize weights to sum to 1
        attn_sum = a24 + a77 + ax
        a24 = a24 / attn_sum
        a77 = a77 / attn_sum
        ax = ax / attn_sum
         # Return as dictionary for easy interpretation
        return {'24GHz': a24, '77GHz': a77, 'Xethru': ax}
    
def train_model(model, train_loader, val_loader, num_epochs=50, patience=10):
    """
    Train the model using the training set and evaluate it on the validation set.
    - model: The model to train.
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - num_epochs: Number of training epochs.
    - patience: Number of epochs to wait for improvement before early stopping.
    - Returns: Training history (loss and accuracy).
    """
    # Training function implements a complete training pipeline with best practices
    
    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Calculate class weights to handle imbalanced data
    # This improves performance when some activities appear more frequently than others

    class_samples = {}
    for _, labels in train_loader:
        for label in labels:
            l = label.item()
            class_samples[l] = class_samples.get(l, 0) + 1
    
    total_samples = sum(class_samples.values())
    # Inverse frequency weighting - rare classes get higher weights
    class_weights = torch.FloatTensor([total_samples / (len(class_samples) * count) 
                                      for count in class_samples.values()])
    
    # Define loss function with class weights to address class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
     #Adam optimizer with weight decay (L2 regularization) to prevent overfitting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay


    # Cosine annealing learning rate scheduler
    # Gradually reduces learning rate following a cosine curve
    # Helps converge to better minima and avoid oscillation
    from torch.optim.lr_scheduler import CosineAnnealingLR
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)        
    
    # Training loop
    # Initialize tracking variables for early stopping and model saving
    best_val_loss = float('inf') #  # Track best validation loss
    counter = 0   # Counter for early stopping patience
    best_val_acc = 0.0 # Track best validation accuracy
    
    # History dictionary to store metrics for later analysis/plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
     # Main training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train() # Set model to training mode (enables dropout, batch norm updates)
        running_loss = 0.0
        correct = 0
        total = 0
        # Iterate through batches in training set
        for inputs, labels in train_loader:
            # Move data to appropriate device
            r24, r77, xethru = [x.to(device) for x in inputs]
            labels = labels.to(device)
            # Zero gradients from previous batch
            optimizer.zero_grad()
            # Forward pass
            outputs = model([r24, r77, xethru])
            # Calculate loss
            loss = criterion(outputs, labels)
             # Backward pass and optimization
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
            
            # Track statistics
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1) # Get predicted class
            total += labels.size(0) # Count total samples
            correct += (predicted == labels).sum().item() # Count correct predictions
        
        # Calculate epoch statistics
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        
        # Validation phase
        model.eval() # Set model to evaluation mode (disables dropout, freezes batch norm)
        running_loss = 0.0
        correct = 0
        total = 0
        
        # No gradient calculation needed for validation
        with torch.no_grad():
            for inputs, labels in val_loader:
                # No gradient calculation needed for validation
                r24, r77, xethru = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # Forward pass only (no backprop during validation)
                outputs = model([r24, r77, xethru])
                loss = criterion(outputs, labels)
                
                # Track statistics
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        # Calculate validation metrics
        epoch_val_loss = running_loss / total
        epoch_val_acc = correct / total
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Update learning rate according to schedule
        scheduler.step()

        # Early stopping check - prevents overfitting by monitoring validation loss
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return history


def evaluate_model(model, test_loader, class_names):
    """
    Evaluate the model on the test set and compute metrics.
    - model: The trained model.
    - test_loader: DataLoader for the test set.
    - class_names: List of activity class names.
    - Returns: Dictionary containing accuracy, confusion matrix, and classification report.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    attention_weights = {'24GHz': [], '77GHz': [], 'Xethru': []}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Get model outputs
            outputs = model([r24, r77, xethru])
            _, preds = torch.max(outputs, 1)
            
            # Get attention weights
            batch_weights = model.get_attention_weights([r24, r77, xethru])
            for k, v in batch_weights.items():
                attention_weights[k].append(v.cpu().numpy())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    # Aggregate attention weights
    for k in attention_weights:
        attention_weights[k] = np.concatenate(attention_weights[k])
    
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
        'attention_weights': attention_weights,
        'class_attention': class_attention
    }


def add_environmental_noise(model, val_loader, noise_levels=[0.01, 0.05, 0.1, 0.2]):
    """Test model under different environmental noise conditions"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                r24, r77, xethru = [x.to(device) for x in inputs]
                labels = labels.to(device)
                
                # Add Gaussian noise to simulate environmental interference
                noisy_r24 = r24 + torch.randn_like(r24) * noise_level
                noisy_r77 = r77 + torch.randn_like(r77) * noise_level
                noisy_xethru = xethru + torch.randn_like(xethru) * noise_level
                
                outputs = model([noisy_r24, noisy_r77, noisy_xethru])
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        results[noise_level] = correct / total
    
    # Plot results
    plt.figure(figsize=(10, 6))
    noise_levels_list = list(results.keys())
    accuracies = [results[n] for n in noise_levels_list]
    
    plt.plot(noise_levels_list, accuracies, marker='o', linestyle='-')
    plt.xlabel('Noise Level')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness to Environmental Noise')
    plt.grid(True)
    plt.savefig('noise_robustness.png')
    plt.show()
    
    return results

# Get activity class names
class_names = dataset.activities


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
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

def plot_radar_importance(class_attention, class_names):
    radars = list(class_attention.keys())
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(14, 8))
    for i, radar in enumerate(radars):
        offset = (i - len(radars)/2 + 0.5) * width
        plt.bar(x + offset, class_attention[radar], width, label=radar)
    
    plt.xlabel('Activity Class')
    plt.ylabel('Average Attention Weight')
    plt.title('Radar Importance by Activity Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('radar_importance.png')
    plt.show()

def analyze_results(results, class_names):
    # Print overall accuracy
    print(f"Overall Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    
    # Plot radar importance by class
    plot_radar_importance(results['class_attention'], class_names)
    
    # Calculate per-radar importance
    radar_importance = {}
    for radar in results['class_attention']:
        radar_importance[radar] = np.mean(results['class_attention'][radar])
    
    print("\nOverall Radar Importance:")
    for radar, importance in radar_importance.items():
        print(f"{radar}: {importance:.4f}")


def analyze_misclassifications(model, val_loader, class_names):
    """Analyze what samples the model misclassifies"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            labels = labels.to(device)
            
            outputs = model([r24, r77, xethru])
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified samples
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    # Get confidence scores
                    confidence = torch.softmax(outputs[i], 0)[pred].item()
                    misclassified.append({
                        'true': class_names[label],
                        'predicted': class_names[pred],
                        'confidence': confidence,
                    })
    
    # Print summary
    if misclassified:
        print(f"Found {len(misclassified)} misclassifications")
        confusion_pairs = {}
        for m in misclassified:
            pair = (m['true'], m['predicted'])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        print("\nConfusion pairs (true->predicted):")
        for (true, pred), count in sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {true} -> {pred}: {count} times")
    else:
        print("No misclassifications found")
    
    return misclassified

def measure_resource_usage(model, input_shape=(1, 3, 224, 224)):
    import time
    import torch
    from thop import profile
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs
    dummy_inputs = [
        torch.randn(input_shape).to(device),
        torch.randn(input_shape).to(device),
        torch.randn(input_shape).to(device)
    ]
    
    # Measure FLOPs and parameters
    macs, params = profile(model, inputs=[dummy_inputs])
    
    # Measure inference time
    warmup_runs = 10
    timing_runs = 100
    
    # Warmup
    for _ in range(warmup_runs):
        _ = model(dummy_inputs)
    
    # Timing
    start_time = time.time()
    for _ in range(timing_runs):
        _ = model(dummy_inputs)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / timing_runs
    
    # Memory usage
    torch.cuda.reset_peak_memory_stats()
    _ = model(dummy_inputs)
    memory_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    return {
        'params': params,
        'macs': macs,
        'inference_time': avg_inference_time * 1000,  # ms
        'memory_usage': memory_usage  # MB
    }

from sklearn.metrics import accuracy_score
def evaluate_radar_contribution(model, test_loader):
    """Evaluate individual radar contributions by ablation study"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    radar_combinations = [
        {'name': 'All Radars', 'mask': [1, 1, 1]},
        {'name': '24GHz Only', 'mask': [1, 0, 0]},
        {'name': '77GHz Only', 'mask': [0, 1, 0]},
        {'name': 'Xethru Only', 'mask': [0, 0, 1]},
        {'name': '24GHz + 77GHz', 'mask': [1, 1, 0]},
        {'name': '24GHz + Xethru', 'mask': [1, 0, 1]},
        {'name': '77GHz + Xethru', 'mask': [0, 1, 1]}
    ]
    
    results = {}
    
    for combo in radar_combinations:
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                r24, r77, xethru = [x.to(device) for x in inputs]
                
                # Apply masking for ablation study
                f24 = model.radar_24g_encoder(r24) if combo['mask'][0] else torch.zeros_like(model.radar_24g_encoder(r24))
                f77 = model.radar_77g_encoder(r77) if combo['mask'][1] else torch.zeros_like(model.radar_77g_encoder(r77))
                fx = model.xethru_encoder(xethru) if combo['mask'][2] else torch.zeros_like(model.xethru_encoder(xethru))
                
                # Fusion and classification
                fused = model.attention_fusion(f24, f77, fx)
                outputs = model.classifier(fused)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        results[combo['name']] = accuracy
    
    return results


# Per-Activity Radar Analysis: Examine which radar type works best for each activity
def plot_radar_activity_heatmap(class_attention, class_names):
    import matplotlib.pyplot as plt
    import numpy as np
    
    data = np.array([class_attention['24GHz'], 
                    class_attention['77GHz'],
                    class_attention['Xethru']])
    
    plt.figure(figsize=(12, 8))
    plt.imshow(data, aspect='auto', cmap='YlGnBu')
    plt.colorbar(label='Attention Weight')
    
    # Add text annotations
    for i in range(len(data)):
        for j in range(len(data[0])):
            plt.text(j, i, f"{data[i][j]:.3f}", 
                     ha="center", va="center", color="black")
    
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(3), ['24GHz', '77GHz', 'Xethru'])
    plt.title('Radar Importance by Activity')
    plt.tight_layout()
    plt.savefig('radar_activity_heatmap.png')
    plt.show()



#  Computational Efficiency Analysis: Add code to measure inference speed and model complexity:
def measure_efficiency(model, val_loader):
    import time
    import torch
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Memory usage (only if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = 0  # Will be updated later
    else:
        memory_used = "N/A - CPU only"
    
    # Model size
    model_size = sum(p.numel() for p in model.parameters()) / 1e6  # in millions
    
    # Get a batch for testing
    try:
        sample_batch = next(iter(val_loader))
        inputs, _ = sample_batch
        inputs = [i.to(device) for i in inputs]
        
        # Warmup
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(inputs)
        
        # Timing inference
        start = time.time()
        with torch.no_grad():
            for _ in range(100):  # Average over 100 runs
                _ = model(inputs)
        avg_time = (time.time() - start) / 100 * 1000  # ms
        
        # Get memory stats if using CUDA
        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
        
    except StopIteration:
        avg_time = "Error: Empty dataloader"
    
    print(f"Model size: {model_size:.2f}M parameters")
    print(f"Memory usage: {memory_used if isinstance(memory_used, str) else f'{memory_used:.2f} MB'}")
    print(f"Average inference time: {avg_time if isinstance(avg_time, str) else f'{avg_time:.2f} ms'}")
    
    return {
        'model_size_M': model_size,
        'memory_usage_MB': memory_used,
        'inference_time_ms': avg_time
    }


def perform_cross_validation(model_class, dataset, k=5, num_epochs=10):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, SubsetRandomSampler
    from sklearn.model_selection import KFold
    
    # Initialize K-Fold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # For storing results
    fold_results = []
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get indices
    indices = list(range(len(dataset)))
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(indices)):
        print(f"FOLD {fold+1}/{k}")
        print("-" * 30)
        
        # Create data samplers
        train_sampler = SubsetRandomSampler(train_ids)
        val_sampler = SubsetRandomSampler(val_ids)
        
        # Create dataloaders
        train_loader = DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=0)
        val_loader = DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=0)
        
        # Initialize model
        model = model_class(num_classes=11)
        model.to(device)
        
        # Define loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
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
                loss = criterion(outputs, labels)
                
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



#  Ablation Studies: Test performance with different combinations of radars:
def ablation_study(model, val_loader, class_names):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
    
    # Determine device
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
        
        # Calculate confusion matrix (optional)
        from sklearn.metrics import confusion_matrix
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
    
    plt.ylim(0, 1.1)  # Set y-axis limit
    plt.tight_layout()
    plt.savefig('radar_combinations_accuracy.png')
    plt.show()
    
    return results, confusion_matrices




#  t-SNE Visualization: Visualize feature space to understand separability
def visualize_features(model, val_loader, class_names):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    features = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            r24, r77, xethru = [x.to(device) for x in inputs]
            
            # Extract features before fusion
            f24 = model.radar_24g_encoder(r24)
            f77 = model.radar_77g_encoder(r77)
            fx = model.xethru_encoder(xethru)
            
            # Get fused features
            fused = model.attention_fusion(f24, f77, fx)
            
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




#   Communication Simulation: Since you mentioned ISAC (Integrated Sensing and Communication), you could simulate communication aspects:
def simulate_communication_constraints(model, val_loader, snr_levels=[30, 20, 10, 5, 0, -5]):
    """Simulate performance under different SNR conditions"""
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for snr in snr_levels:
        # Function to add noise based on SNR
        def add_noise(signal, snr_db):
            signal_power = torch.mean(signal**2)
            # Avoid division by zero
            if signal_power < 1e-10:
                return signal  
            noise_power = signal_power / (10**(snr_db/10))
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
    
    # Plot results
    plt.figure(figsize=(10, 6))
    snrs = list(results.keys())
    accuracies = [results[s] for s in snrs]
    
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
    
    return results

def quantize_model(model):
    import torch
    
    # Make a copy of the model for quantization
    quantized_model = copy.deepcopy(model)
    
    # Set model to evaluation mode
    quantized_model.eval()
    
    # Fuse Conv, BN, Relu layers if applicable
    # This step depends on your specific model architecture
    
    # Specify quantization configuration
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    
    return quantized_model

def compare_model_sizes(original_model, quantized_model):
    import os
    import torch
    
    # Save the original model
    torch.save(original_model.state_dict(), "original_model.pth")
    original_size = os.path.getsize("original_model.pth") / (1024 * 1024)
    
    # Save the quantized model
    torch.save(quantized_model.state_dict(), "quantized_model.pth")
    quantized_size = os.path.getsize("quantized_model.pth") / (1024 * 1024)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")
    
    return {'original_size_MB': original_size, 
            'quantized_size_MB': quantized_size,
            'reduction_percentage': (1 - quantized_size/original_size)*100}




if __name__ == '__main__':
    import multiprocessing
    mp.freeze_support()  # Add this line
    
    print("Setting up datasets and data loaders...")
    # Use the datasets created earlier with proper transformations
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    print("Initializing model...")
    model = MultiRadarFusion(num_classes=11)
    
    print("Training model with improved settings...")
    history = train_model(model, train_loader, val_loader, num_epochs=50, patience=10)
    
    print("Loading best model...")
    best_model = MultiRadarFusion(num_classes=11)
    best_model.load_state_dict(torch.load('best_model.pth'))
    
    # Prepare data loaders
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Reduce num_workers for Windows compatibility
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Check if trained model exists, otherwise train it
    import os
    if not os.path.exists('best_model.pth'):
        print("Training model from scratch...")
        history = train_model(model, train_loader, val_loader, num_epochs=50)
        print("Model training completed and saved as best_model.pth")
    else:
        print("Found pre-trained model. Loading...")
    
    
    # Continue with evaluation
    print("Evaluating model...")
    results = evaluate_model(best_model, test_loader, class_names)
    analyze_results(results, class_names)
    
    # Measure resources
    resource_metrics = measure_resource_usage(best_model)
    print(f"Model Parameters: {resource_metrics['params'] / 1e6:.2f}M")
    print(f"Computational Complexity: {resource_metrics['macs'] / 1e9:.2f}G MACs")
    print(f"Inference Time: {resource_metrics['inference_time']:.2f} ms")
    print(f"GPU Memory Usage: {resource_metrics['memory_usage']:.2f} MB")
    
    # Run additional analyses
    radar_contributions = evaluate_radar_contribution(best_model, test_loader)
    efficiency_metrics = measure_efficiency(best_model, val_loader)
    ablation_results, ablation_cms = ablation_study(best_model, val_loader, class_names)
    visualize_features(best_model, val_loader, class_names)
    
    # Plot radar importance
    plot_radar_activity_heatmap(results['class_attention'], class_names)
    
    # Plot radar contribution results
    plt.figure(figsize=(10, 6))
    names = list(radar_contributions.keys())
    accuracies = list(radar_contributions.values())
    plt.bar(names, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Radar Combination')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('radar_combinations.png')
    plt.show()


    # Run noise robustness test
    print("Testing model robustness to noise...")
    noise_results = add_environmental_noise(best_model, val_loader)
    
    # Analyze misclassifications
    print("Analyzing misclassifications...")
    misclassifications = analyze_misclassifications(best_model, val_loader, class_names)
    
    # Perform proper cross-validation
    print("Performing 5-fold cross-validation...")
    cv_results, cv_avg, cv_std = perform_cross_validation(MultiRadarFusion, dataset, k=5, num_epochs=20)
    print(f"Cross-validation accuracy: {cv_avg:.4f} ± {cv_std:.4f}")
    
    print("Analysis complete!")
    
    # Optional: run cross-validation (this will take time)
    # cv_results, cv_avg, cv_std = perform_cross_validation(MultiRadarFusion, dataset, k=5, num_epochs=10)
    
    # Optional: communication simulation
    # communication_results = simulate_communication_constraints(best_model, val_loader)
    
    # Optional: model quantization
    # import copy
    # quantized_model = quantize_model(best_model.cpu())
    # size_comparison = compare_model_sizes(best_model.cpu(), quantized_model)