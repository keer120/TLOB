import random
from lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from utils.utils_model import pick_model
import constants as cst
from scipy.stats import mode


class Engine(LightningModule):
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        num_features,
        dataset_type,
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        is_sin_emb=True,
        len_test_dataloader=None,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.model_type = model_type
        self.num_heads = num_heads
        self.is_wandb = is_wandb
        self.len_test_dataloader = len_test_dataloader
        self.lr = lr
        self.optimizer = optimizer
        self.dir_ckpt = dir_ckpt
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.experiment_type = experiment_type
        self.model = pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type) 
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        self.class_weights = None
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_targets = []
        self.test_predictions = []
        self.test_proba = []
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.min_loss = np.inf
        self.save_hyperparameters()
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
        
    def forward(self, x, batch_idx=None):
        output = self.model(x)
        return output
    
    @property
    def loss_function(self):
        if self.class_weights is not None:
            return nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            return nn.CrossEntropyLoss()

    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        # Compute class weights on first batch if not set
        if self.class_weights is None:
            class_sample_count = torch.bincount(y)
            class_weights = 1. / class_sample_count.float()
            class_weights = class_weights / class_weights.sum() * len(class_sample_count)
            self.class_weights = class_weights.to(y.device)
            print(f"[Engine] Using class weights: {self.class_weights}")
        y_hat = self.forward(x)
        batch_loss = self.loss(y_hat, y)
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f'train loss: {sum(self.train_losses) / len(self.train_losses)}')
        return batch_loss_mean
    
    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Validation: with EMA
        with self.ema.average_parameters():
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)
            self.val_targets.append(y.cpu().numpy())
            self.val_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_losses.append(batch_loss_mean.item())
        return batch_loss_mean
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)
        y_hat = self.forward(x, batch_idx)
        # Debug: check for NaN/Inf in y_hat and y
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            print(f"[DEBUG] NaN or Inf in y_hat in test_step, batch {batch_idx}")
        if torch.isnan(y).any() or torch.isinf(y).any():
            print(f"[DEBUG] NaN or Inf in y in test_step, batch {batch_idx}")
        batch_loss = self.loss(y_hat, y)
        batch_loss_mean = torch.mean(batch_loss)
        if torch.isnan(batch_loss_mean) or torch.isinf(batch_loss_mean):
            print(f"[DEBUG] Skipping batch {batch_idx} due to NaN/Inf in batch_loss_mean")
            if not hasattr(self, 'skipped_test_batches'):
                self.skipped_test_batches = 0
            self.skipped_test_batches += 1
            return None  # Do not append anything for this batch
        else:
            self.test_losses.append(batch_loss_mean.item())
            self.test_targets.append(y.cpu().numpy())
            self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
        return batch_loss_mean
    
    def on_validation_epoch_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.train_losses = []
        # Store train loss for combined plotting
        self.current_train_loss = loss
        print(f'Train loss on epoch {self.current_epoch}: {loss}')
        
    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []
        
        # model checkpointing
        if self.val_loss < self.min_loss:
            # if the improvement is less than 0.0002, we halve the learning rate
            if self.val_loss - self.min_loss > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2  
            self.min_loss = self.val_loss
            self.model_checkpointing(self.val_loss)
        else:
            self.optimizer.param_groups[0]["lr"] /= 2
        
        # Log losses to wandb (both individually and in the same plot)
        self.log_losses_to_wandb(self.current_train_loss, self.val_loss)
        
        # Continue with regular Lightning logging for compatibility
        self.log("val_loss", self.val_loss)
        print(f'Validation loss on epoch {self.current_epoch}: {self.val_loss}')
        targets = np.concatenate(self.val_targets)    
        predictions = np.concatenate(self.val_predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("val_f1_score", class_report["macro avg"]["f1-score"])
        self.log("val_accuracy", class_report["accuracy"])
        self.log("val_precision", class_report["macro avg"]["precision"])
        self.log("val_recall", class_report["macro avg"]["recall"])
        self.val_targets = []
        self.val_predictions = [] 
    
    def log_losses_to_wandb(self, train_loss, val_loss):
        """Log training and validation losses to wandb in the same plot."""
        if self.is_wandb:   
            # Log combined losses for a single plot
            wandb.log({
                "losses": {
                    "train": train_loss,
                    "validation": val_loss
                },
                "epoch": self.global_step
            })
    
    def on_test_epoch_end(self) -> None:
        targets = np.concatenate(self.test_targets)    
        predictions = np.concatenate(self.test_predictions)
        predictions_path = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "predictions")
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        np.save(predictions_path, predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        # Debug: check if test_losses is empty
        if len(self.test_losses) == 0:
            print("[DEBUG] test_losses is empty in on_test_epoch_end!")
            test_loss = float('nan')
        else:
            test_loss = sum(self.test_losses) / len(self.test_losses)
        self.log("test_loss", test_loss)
        if hasattr(self, 'skipped_test_batches'):
            print(f"[DEBUG] Skipped {self.skipped_test_batches} test batches due to NaN/Inf loss.")
            del self.skipped_test_batches
        self.log("f1_score", class_report["macro avg"]["f1-score"])
        self.log("accuracy", class_report["accuracy"])
        self.log("precision", class_report["macro avg"]["precision"])
        self.log("recall", class_report["macro avg"]["recall"])
        # Only plot confusion matrix during evaluation or fine-tuning, not training
        if getattr(self, 'experiment_type', None) in ["EVALUATION", "FINETUNING"]:
            try:
                from preprocessing.sbi import save_confusion_matrix
                cm_path = os.path.join(os.path.dirname(predictions_path), "confusion_matrix.png")
                save_confusion_matrix(targets, predictions, cm_path)
            except Exception as e:
                print(f"Could not plot confusion matrix: {e}")
        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []  
        self.first_test = False
        test_proba = np.concatenate(self.test_proba)
        if np.isnan(test_proba).any():
            print("Warning: test_proba contains NaN values. Skipping precision-recall curve plot.")
        else:
            precision, recall, _ = precision_recall_curve(targets, test_proba, pos_label=1)
            self.plot_pr_curves(recall, precision, self.is_wandb)
        
    def configure_optimizers(self):
        if self.model_type == "DEEPLOB":
            eps = 1
        else:
            eps = 1e-8
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'Lion':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer
    
    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")

    def model_checkpointing(self, loss):        
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        filename_ckpt = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".pt"
                             )
        path_ckpt = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "pt", filename_ckpt)
        
        # Save PyTorch checkpoint
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt)
            
            # Save ONNX model
            onnx_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            
            onnx_filename = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".onnx"
                            )
            onnx_path = os.path.join(onnx_dir, onnx_filename)
            
            # Create dummy input with appropriate shape
            dummy_input = torch.randn(1, self.seq_size, self.num_features, device=self.device)
            
            # Export to ONNX
            try:
                torch.onnx.export(
                    self.model,                  # model being run
                    dummy_input,                 # model input (or a tuple for multiple inputs)
                    onnx_path,                   # where to save the model
                    export_params=True,          # store the trained parameter weights inside the model file
                    opset_version=12,            # the ONNX version to export the model to
                    do_constant_folding=True,    # whether to execute constant folding for optimization
                    input_names=['input'],       # the model's input names
                    output_names=['output'],     # the model's output names
                    dynamic_axes={               # variable length axes
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            except Exception as e:
                print(f"Failed to export ONNX model: {e}")
        
        self.last_path_ckpt = path_ckpt  
        
    def plot_pr_curves(self, recall, precision, is_wandb):
        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(recall, precision, label='Precision-Recall', color='black')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        if is_wandb:
            wandb.log({f"precision_recall_curve_{self.dataset_type}": wandb.Image(plt)})
        plt.savefig(cst.DIR_SAVED_MODEL + "/" + str(self.model_type) + "/" +f"precision_recall_curve_{self.dataset_type}.svg")
        #plt.show()
        plt.close()
        
def compute_most_attended(att_feature):
    ''' att_feature: list of tensors of shape (num_samples, num_layers, 2, num_heads, num_features) '''
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)  # Use transpose instead of permute
    ''' att_feature: shape (num_layers, num_heads, num_samples, 2, num_features) '''
    indices = att_feature[:, :, :, 1]
    values = att_feature[:, :, :, 0]
    most_frequent_indices = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]), dtype=int)
    average_values = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]))
    for layer in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            for seq in range(indices.shape[3]):
                # Extract the indices for the current layer and sequence element
                current_indices = indices[layer, head, :, seq]
                current_values = values[layer, head, :, seq]
                # Find the most frequent index
                most_frequent_index = mode(current_indices, keepdims=False)[0]
                # Store the result
                most_frequent_indices[layer, head, seq] = most_frequent_index
                # Compute the average value for the most frequent index
                avg_value = np.mean(current_values[current_indices == most_frequent_index])
                # Store the average value
                average_values[layer, head, seq] = avg_value
    return most_frequent_indices, average_values



