import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import seaborn as sns
from tqdm import tqdm
import copy
import pandas as pd

class ImageJudgmentSystem:
    def __init__(self, num_classes, model_name='resnet50', pretrained=True, fine_tuning_mode='full'):
        """
        Initialize the Image Judgment System with fine-tuning support.
        
        Args:
            num_classes (int): Number of classes for classification
            model_name (str): ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained (bool): Whether to use pre-trained weights
            fine_tuning_mode (str): Fine-tuning approach:
                - 'full': Fine-tune the entire network
                - 'last_layer': Only train the final layer
                - 'last_block': Fine-tune only the last residual block and final layer
                - 'progressive': Gradually unfreeze layers during training
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained
        self.fine_tuning_mode = fine_tuning_mode
        
        # Define transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self._initialize_model()
        
        # Initialize other variables
        self.class_names = None
        self.image_paths = None
        self.labels = None
        self.best_val_acc = 0.0
        self.current_unfreeze_epoch = 0  # For progressive fine-tuning
    
    def _initialize_model(self):
        """Initialize the ResNet model with specified parameters and set up for fine-tuning"""
        if self.model_name == 'resnet18':
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            self.model = models.resnet34(pretrained=self.pretrained)
        elif self.model_name == 'resnet50':
            self.model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'resnet101':
            self.model = models.resnet101(pretrained=self.pretrained)
        elif self.model_name == 'resnet152':
            self.model = models.resnet152(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Modify final fully connected layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        # Set up fine-tuning mode
        self._setup_fine_tuning()
        
        self.model = self.model.to(self.device)
        
        # Initialize optimizer and loss function
        if self.fine_tuning_mode == 'last_layer' or self.fine_tuning_mode == 'last_block':
            # Only optimize parameters that require gradients
            params_to_update = [p for p in self.model.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(params_to_update, lr=0.001)
        else:
            # Use different learning rates for different parts of the model
            if self.fine_tuning_mode == 'full':
                # Lower learning rate for pre-trained parameters
                backbone_params = {'params': [p for n, p in self.model.named_parameters() if 'fc' not in n], 'lr': 0.0001}
                fc_params = {'params': self.model.fc.parameters(), 'lr': 0.001}
                self.optimizer = optim.Adam([backbone_params, fc_params])
            else:  # progressive
                # Start with only training the FC layer
                self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3, factor=0.5)
    
    def _setup_fine_tuning(self):
        """Configure model layers for different fine-tuning approaches"""
        if self.fine_tuning_mode == 'last_layer':
            # Freeze all layers except the final layer
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze the final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
            print("Fine-tuning mode: Training only the final layer.")
            
        elif self.fine_tuning_mode == 'last_block':
            # Freeze all layers
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Unfreeze the final layer and the last residual block
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
            # For ResNet models, layer4 is the last block
            for param in self.model.layer4.parameters():
                param.requires_grad = True
                
            print("Fine-tuning mode: Training the last residual block and final layer.")
            
        elif self.fine_tuning_mode == 'progressive':
            # Start by freezing all layers except the final layer
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Unfreeze the final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
                
            print("Fine-tuning mode: Progressive unfreezing starting with final layer.")
            
        elif self.fine_tuning_mode == 'full':
            # All parameters are trainable by default
            for param in self.model.parameters():
                param.requires_grad = True
                
            print("Fine-tuning mode: Training all layers.")
            
        else:
            raise ValueError(f"Unsupported fine-tuning mode: {self.fine_tuning_mode}")
            
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    def _update_progressive_fine_tuning(self, epoch):
        """Progressively unfreeze more layers as training progresses"""
        if self.fine_tuning_mode != 'progressive':
            return
            
        # Only update if we're at a new epoch that needs unfreezing
        if epoch <= self.current_unfreeze_epoch:
            return
            
        self.current_unfreeze_epoch = epoch
        
        # Unfreeze layers based on epoch number
        if epoch == 3:  # After 3 epochs, unfreeze layer4
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            print("Epoch 3: Unfreezing layer4")
            
            # Update optimizer to include newly unfrozen parameters
            self.optimizer = optim.Adam([
                {'params': self.model.fc.parameters(), 'lr': 0.001},
                {'params': self.model.layer4.parameters(), 'lr': 0.0001}
            ])
            
        elif epoch == 6:  # After 6 epochs, unfreeze layer3
            for param in self.model.layer3.parameters():
                param.requires_grad = True
            print("Epoch 6: Unfreezing layer3")
            
            # Update optimizer to include newly unfrozen parameters
            self.optimizer = optim.Adam([
                {'params': self.model.fc.parameters(), 'lr': 0.001},
                {'params': self.model.layer4.parameters(), 'lr': 0.0001},
                {'params': self.model.layer3.parameters(), 'lr': 0.00005}
            ])
            
        elif epoch == 9:  # After 9 epochs, unfreeze remaining layers
            for param in self.model.parameters():
                param.requires_grad = True
            print("Epoch 9: Unfreezing all remaining layers")
            
            # Update optimizer with different learning rates for different parts
            self.optimizer = optim.Adam([
                {'params': self.model.fc.parameters(), 'lr': 0.001},
                {'params': self.model.layer4.parameters(), 'lr': 0.0001},
                {'params': self.model.layer3.parameters(), 'lr': 0.00005},
                {'params': [p for n, p in self.model.named_parameters() 
                           if not any(x in n for x in ['fc', 'layer3', 'layer4'])], 'lr': 0.00001}
            ])
            
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    
    class CustomImageDataset(Dataset):
        """Custom Dataset for loading images"""
        def __init__(self, image_paths, labels, transform=None, class_names=None):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
            self.class_names = class_names
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
    
    def load_data(self, data_dir):
        """
        Load data from directory structure.
        
        Args:
            data_dir (str): Path to data directory
        
        Directory structure should be:
        data_dir/
            ├── class1/
            │   ├── img1.jpg
            │   └── img2.jpg
            └── class2/
                ├── img3.jpg
                └── img4.jpg
        """
        # Get class names from directory
        self.class_names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
        
        print(f"Loaded {len(self.image_paths)} images.")
        print(f"Classes: {self.class_names}")
    
    def create_data_loaders(self, train_indices, val_indices, batch_size=16):
        """
        Create data loaders for training and validation sets.
        
        Args:
            train_indices (list): Indices for training set
            val_indices (list): Indices for validation set
            batch_size (int): Batch size for data loaders
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        # Create samplers
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        # Create datasets
        train_dataset = self.CustomImageDataset(
            self.image_paths, self.labels, self.train_transform, self.class_names
        )
        val_dataset = self.CustomImageDataset(
            self.image_paths, self.labels, self.test_transform, self.class_names
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4
        )
        
        return train_loader, val_loader
    
    def train_fold(self, train_loader, val_loader, epochs=10, save_dir='models', fold=None):
        """
        Train the model for a single fold.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of training epochs
            save_dir (str): Directory to save model checkpoints
            fold (int, optional): Current fold number
            
        Returns:
            tuple: (best_model_state_dict, history, best_val_acc)
        """
        # Reset model, optimizer, and scheduler for each fold
        self._initialize_model()
        self.current_unfreeze_epoch = 0  # Reset progressive unfreezing counter
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        best_model_state_dict = None
        fold_str = f" [Fold {fold}]" if fold is not None else ""
        
        for epoch in range(epochs):
            # Update progressive fine-tuning if needed
            if self.fine_tuning_mode == 'progressive':
                self._update_progressive_fine_tuning(epoch)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}{fold_str} [Train]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}{fold_str} [Val]")
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Get current learning rates
            current_lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
            avg_lr = sum(current_lrs) / len(current_lrs)
            
            # Save checkpoint if validation accuracy improved
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                
                if fold is not None:
                    model_path = os.path.join(save_dir, f'best_model_fold_{fold}.pth')
                else:
                    model_path = os.path.join(save_dir, 'best_model.pth')
                    
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'class_names': self.class_names,
                    'fine_tuning_mode': self.fine_tuning_mode
                }, model_path)
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            
            # Print epoch results
            print(f"Epoch {epoch+1}/{epochs}{fold_str}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate(s): {current_lrs}")
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(avg_lr)
        
        return best_model_state_dict, history, best_val_acc
    
    def double_cross_validation(self, outer_folds=5, inner_folds=3, epochs=10, batch_size=16, save_dir='models'):
        """
        Perform double cross-validation.
        
        Args:
            outer_folds (int): Number of folds for outer CV loop
            inner_folds (int): Number of folds for inner CV loop
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            save_dir (str): Directory to save model checkpoints
            
        Returns:
            dict: Results of double cross-validation
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Set up outer cross-validation
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
        
        # Lists to store results
        outer_fold_results = []
        best_inner_models = []
        best_outer_models = []
        
        # Outer cross-validation loop
        for outer_fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(self.image_paths)):
            print(f"\n===== Outer Fold {outer_fold+1}/{outer_folds} =====")
            print(f"Fine-tuning mode: {self.fine_tuning_mode}")
            
            # Set up inner cross-validation
            inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
            
            # Lists to store inner results
            inner_fold_results = []
            inner_fold_models = []
            
            # Inner cross-validation loop
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(train_val_idx)):
                print(f"\n----- Inner Fold {inner_fold+1}/{inner_folds} -----")
                
                # Map indices back to original dataset indices
                inner_train_indices = [train_val_idx[i] for i in inner_train_idx]
                inner_val_indices = [train_val_idx[i] for i in inner_val_idx]
                
                # Create data loaders
                inner_train_loader, inner_val_loader = self.create_data_loaders(
                    inner_train_indices, inner_val_indices, batch_size
                )
                
                # Train model on inner fold
                best_model_state, history, best_val_acc = self.train_fold(
                    inner_train_loader, inner_val_loader, epochs, save_dir, 
                    fold=f"outer_{outer_fold+1}_inner_{inner_fold+1}"
                )
                
                # Store results
                inner_fold_results.append({
                    'outer_fold': outer_fold + 1,
                    'inner_fold': inner_fold + 1,
                    'val_acc': best_val_acc,
                    'history': history
                })
                
                inner_fold_models.append({
                    'state_dict': best_model_state,
                    'val_acc': best_val_acc
                })
            
            # Find best model from inner folds
            best_inner_idx = np.argmax([model['val_acc'] for model in inner_fold_models])
            best_inner_model = inner_fold_models[best_inner_idx]
            best_inner_models.append(best_inner_model)
            
            print(f"\nBest inner model from fold {best_inner_idx+1} with validation accuracy: {best_inner_model['val_acc']:.4f}")
            
            # Train final model on all training data with the best hyperparameters
            print("\n----- Training final model on all training data -----")
            train_loader, test_loader = self.create_data_loaders(
                train_val_idx, test_idx, batch_size
            )
            
            # Load the best model from inner CV
            self.model.load_state_dict(best_inner_model['state_dict'])
            
            # Evaluate on test set
            test_acc, test_confusion, test_report = self.evaluate(
                test_loader, fold=f"outer_{outer_fold+1}_final"
            )
            
            # Store outer fold results
            outer_fold_results.append({
                'outer_fold': outer_fold + 1,
                'test_acc': test_acc,
                'confusion_matrix': test_confusion,
                'classification_report': test_report
            })
            
            # Save the final model for this outer fold
            outer_model_path = os.path.join(save_dir, f'final_model_outer_fold_{outer_fold+1}.pth')
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'val_acc': best_inner_model['val_acc'],
                'test_acc': test_acc,
                'class_names': self.class_names,
                'fine_tuning_mode': self.fine_tuning_mode
            }, outer_model_path)
            
            best_outer_models.append({
                'outer_fold': outer_fold + 1,
                'state_dict': copy.deepcopy(self.model.state_dict()),
                'test_acc': test_acc
            })
        
        # Calculate overall performance across all outer folds
        test_accs = [result['test_acc'] for result in outer_fold_results]
        mean_test_acc = np.mean(test_accs)
        std_test_acc = np.std(test_accs)
        
        print("\n===== Double Cross-Validation Results =====")
        print(f"Fine-tuning mode: {self.fine_tuning_mode}")
        print(f"Mean Test Accuracy: {mean_test_acc:.4f} ± {std_test_acc:.4f}")
        
        # Find best outer fold model
        best_outer_idx = np.argmax([model['test_acc'] for model in best_outer_models])
        best_outer_model = best_outer_models[best_outer_idx]
        
        print(f"Best model from outer fold {best_outer_idx+1} with test accuracy: {best_outer_model['test_acc']:.4f}")
        
        # Save final best model
        self.model.load_state_dict(best_outer_model['state_dict'])
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'test_acc': best_outer_model['test_acc'],
            'class_names': self.class_names,
            'fine_tuning_mode': self.fine_tuning_mode
        }, os.path.join(save_dir, 'best_final_model.pth'))
        
        # Return comprehensive results
        return {
            'outer_results': outer_fold_results,
            'inner_results': inner_fold_results,
            'mean_test_acc': mean_test_acc,
            'std_test_acc': std_test_acc,
            'best_model_fold': best_outer_idx + 1,
            'best_model_acc': best_outer_model['test_acc'],
            'fine_tuning_mode': self.fine_tuning_mode
        }
    
    def evaluate(self, test_loader, fold=None):
        """
        Evaluate the model on test data.
        
        Args:
            test_loader (DataLoader): Test data loader
            fold (str, optional): Fold identifier for saving confusion matrix
        
        Returns:
            tuple: (accuracy, confusion matrix, classification report)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        accuracy = test_correct / test_total
        conf_matrix = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        fold_str = f" - Fold {fold}" if fold else ""
        mode_str = f" - {self.fine_tuning_mode.replace('_', ' ').title()}"
        plt.title(f'Confusion Matrix{fold_str}{mode_str}')
        plt.tight_layout()
        
        if fold:
            plt.savefig(f'confusion_matrix_{fold}_{self.fine_tuning_mode}.png')
        
        plt.show()
        
        return accuracy, conf_matrix, report
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to saved model
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.class_names = checkpoint.get('class_names')
        loaded_fine_tuning_mode = checkpoint.get('fine_tuning_mode')
        
        print(f"Loaded model from {model_path}")
        if self.class_names:
            print(f"Classes: {self.class_names}")
        if loaded_fine_tuning_mode:
            print(f"Model was trained with fine-tuning mode: {loaded_fine_tuning_mode}")
    
    def predict_image(self, image_path):
        """
        Predict class for a single image.
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (predicted class name, confidence score, all class probabilities)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.test_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[prediction.item()]
        confidence_score = confidence.item()
        all_probs = probabilities.squeeze().cpu().numpy()
        
        # Create a dictionary mapping class names to probabilities
        class_probs = {cls: prob for cls, prob in zip(self.class_names, all_probs)}
        
        return predicted_class, confidence_score, class_probs
    
    def visualize_predictions(self, image_paths, figsize=(15, 10), show_all_probs=False):
        """
        Visualize predictions for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            