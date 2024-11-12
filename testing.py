import os
import torch
import segmentation_models_pytorch as smp  # Import the segmentation_models.pytorch library
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Custom Dataset (same as training code)
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_mask.png'))
            mask = Image.open(mask_path).convert('L')
            mask = transforms.ToTensor()(mask)
            if self.transform:
                image = self.transform(image)
            return image, mask, self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.images[idx]

# Evaluation function
def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_masks = []
    
    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = (outputs > threshold).float()
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_masks.extend(masks.cpu().numpy().flatten())
    
    accuracy = accuracy_score(all_masks, all_preds)
    precision = precision_score(all_masks, all_preds, zero_division=0)
    recall = recall_score(all_masks, all_preds, zero_division=0)
    f1 = f1_score(all_masks, all_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Function to save predicted masks
def save_predictions(model, test_loader, output_dir, device, threshold=0.5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for images, masks, filenames in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = (outputs > threshold).float()
            
            for img, mask, pred, fname in zip(images, masks, preds, filenames):
                img = img.cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                mask = mask.cpu().numpy().squeeze()
                pred = pred.cpu().numpy().squeeze()
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{os.path.splitext(fname)[0]}_result.png'))
                plt.close()

# Testing function to evaluate models
def test_models(model_dir, data_dir, output_dir, device):
    transform = transforms.Compose([ 
        transforms.Resize((256, 256)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    model_types = ['stroma', 'tumor', 'til']
    results = {}
    
    for model_type in model_types:
        print(f'\nTesting {model_type} model...')
        
        # Initialize the model with the same architecture as the training code
        model = smp.DeepLabV3(
            encoder_name="resnet50",        # Encoder architecture (ResNet50 used here)
            encoder_weights="imagenet",     # Pretrained weights for the encoder
            in_channels=3,                  # Number of input channels (3 for RGB images)
            classes=1                       # Number of output classes (1 for binary segmentation)
        ).to(device)
        
        # Load the model weights
        model.load_state_dict(torch.load(os.path.join(model_dir, f'{model_type}_model.pth')))
        model.eval()
        
        test_dir = os.path.join(data_dir, model_type, 'test')
        test_dataset = SegmentationDataset(
            os.path.join(test_dir, 'images'),
            os.path.join(test_dir, 'masks'),
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Evaluate the model
        metrics = evaluate_model(model, test_loader, device)
        results[model_type] = metrics
        
        # Save the predictions
        save_predictions(model, test_loader, os.path.join(output_dir, f'{model_type}_predictions'), device)
        
        print(f'Results for {model_type}:')
        for metric, value in metrics.items():
            print(f'{metric}: {value:.4f}')
    
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_dir = '/mnt/storage7/code/code/segmentation_models.pytorch/segmentation_models_pytorch/out5'  # Adjust paths as needed
    data_dir = '/mnt/storage7/code/code/segmentation_models.pytorch/segmentation_models_pytorch/datasets_folders'
    output_dir = '/mnt/storage7/code/code/segmentation_models.pytorch/segmentation_models_pytorch/test_results2'
    
    # Run the testing function
    results = test_models(model_dir=model_dir, data_dir=data_dir, output_dir=output_dir, device=device)
