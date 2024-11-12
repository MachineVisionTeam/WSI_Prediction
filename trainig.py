import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp  # Import the segmentation_models.pytorch library
from PIL import Image
import torch
import torch.nn as nn
# Custom Dataset
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize mask to match image size
        if image.size != mask.size:
            mask = mask.resize(image.size)
        
        # Convert mask to tensor (no binary threshold for multi-class)
        mask = transforms.ToTensor()(mask)
        
        if self.transform:
            image = self.transform(image)
        
        return image, mask.float()

# Custom loss function
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, output, target):
        smooth = 1.
        output = torch.sigmoid(output)
        intersection = (output * target).sum()
        union = output.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

# Training function
def train_model(data_dir, output_dir, device, num_epochs=100, patience=10):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Loop over each model type
    for model_type in ['stroma', 'tumor', 'til']:
        print(f'\nTraining {model_type} model...')
        
        # Define dataset paths
        train_dir = os.path.join(data_dir, model_type, 'train')
        val_dir = os.path.join(data_dir, model_type, 'val')
        
        # Create datasets
        train_dataset = SegmentationDataset(
            os.path.join(train_dir, 'images'),
            os.path.join(train_dir, 'masks'),
            transform=transform
        )
        
        val_dataset = SegmentationDataset(
            os.path.join(val_dir, 'images'),
            os.path.join(val_dir, 'masks'),
            transform=transform
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Initialize the UNet model from segmentation_models.pytorch
        model = smp.DeepLabV3(
            encoder_name="resnet50",        
            encoder_weights="imagenet",     # Pretrained weights for the encoder
            in_channels=3,                  # Number of input channels (3 for RGB images)
            classes=1                       # Number of output classes (1 for binary, or adjust for multi-class)
        ).to(device)
        
        # Loss function and optimizer
        criterion = DiceLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
            
            # Print progress
            print(f'Epoch {epoch+1}/{num_epochs}')
            print(f'Training Loss: {train_loss/len(train_loader):.4f}')
            print(f'Validation Loss: {val_loss/len(val_loader):.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, f'{model_type}_model.pth'))
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping...')
                    break

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
data_dir = '/mnt/storage7/code/code/segmentation_models.pytorch/segmentation_models_pytorch/datasets_folders'
output_dir = '/mnt/storage7/code/code/segmentation_models.pytorch/segmentation_models_pytorch/out5'

# Train models
train_model(data_dir, output_dir, device)
