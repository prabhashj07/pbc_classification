import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import datetime
import os
import numpy as np

from src.models.factory import ModelFactory
from src.early_stopping import EarlyStopping
from src.utils import calculate_sensitivity_specivity
from src.dataset import create_dataloaders  

# Function to get current timestamp in a desired format
def get_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


# Set random seed for reproducibility
def set_seed(seed: int):
    """
    Set the seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU support
    np.random.seed(seed)
    # If using deterministic operations on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification model.")
    parser.add_argument("--model_name", type=str, default='vit_base_patch16_224', help="Name of the model.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--use_scheduler", action='store_true', help="Use learning rate scheduler.")
    return parser.parse_args()

class ClassificationModel:
    def __init__(self, model_name, num_classes, lr=2e-4, use_scheduler=True, device='cuda'):
        self.device = device
        self.model = ModelFactory(model_name, num_classes)().to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        ) if use_scheduler else None
        
        # Set up the log file
        self.log_dir = './artifacts/logs' 
        os.makedirs(self.log_dir, exist_ok=True) 
        timestamp = get_timestamp().replace(" ", "_").replace(":", "-")
        self.log_file = os.path.join(self.log_dir, f'training_log_{timestamp}.txt')
        self._log(f"Training started with model: {model_name}")

    def _log(self, message):
        timestamp = get_timestamp()
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp} - {message}\n")
        print(f"{timestamp} - {message}")  

    def train(self, train_loader, val_loader, epochs, early_stopping_patience=5, early_stopping_delta=0.001):
        early_stopping = EarlyStopping(patience=early_stopping_patience, delta=early_stopping_delta, verbose=True)
        self._log(f"Training started with learning rate = {self.optimizer.param_groups[0]['lr']}")

        for epoch in range(epochs):
            self._log(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress = int((batch_idx + 1) / len(train_loader) * 100)
                if (batch_idx + 1) % (len(train_loader) // 20 + 1) == 0 or batch_idx == len(train_loader) - 1:
                    self._log(f"Training: {progress}% | Loss: {running_loss / total:.4f} | Acc: {correct / total:.4f}")
            
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            self._log(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_metrics[0]:.4f} | Recall: {val_metrics[1]:.4f} | F1: {val_metrics[2]:.4f}")

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                self._log(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if self.scheduler:
                self.scheduler.step(val_loss)

    def validate(self, val_loader):
        self.model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels, all_preds = [], []

        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
                progress = int((batch_idx + 1) / len(val_loader) * 100)
                if (batch_idx + 1) % (len(val_loader) // 10 + 1) == 0 or batch_idx == len(val_loader) - 1:
                    self._log(f"Validation: {progress}% | Loss: {val_loss / val_total:.4f} | Acc: {val_correct / val_total:.4f}")

        val_loss /= val_total
        val_acc = val_correct / val_total
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        sensitivity, specificity = calculate_sensitivity_specivity(all_labels, all_preds, len(set(all_labels)))

        return val_loss, val_acc, (precision, recall, f1, sensitivity, specificity)


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)

    args = parse_args()
    dataset_dir = '../data/PBC_dataset_normal_DIB_224/PBC_dataset_normal_DIB_224'
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(dataset_dir, batch_size=args.batch_size)
    model = ClassificationModel(args.model_name, num_classes=8, lr=args.lr, use_scheduler=args.use_scheduler)
    model.train(train_loader, val_loader, args.epochs)
    test_loss, test_acc, test_metrics = model.validate(test_loader)
    print(f"Test | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Precision: {test_metrics[0]:.4f} | Recall: {test_metrics[1]:.4f} | F1: {test_metrics[2]:.4f}")

