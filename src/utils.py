import torch
import random
import numpy as np
import tarfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=777):
    seed = seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)



def gzip_folder(folder_path, output_file):
    """
    Compresses an entire folder into a single .tar.gz file.
    
    Args:
        folder_path (str): Path to the folder to compress.
        output_file (str): Path to the output .tar.gz file.
    """
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    print(f"Folder '{folder_path}' has been compressed into '{output_file}'")

# Example usage
# folder_path = "./testfolder/submission"            # Path to the folder you want to compress
# output_file = "./testfolder/submission.gz"         # Output .gz file name
# gzip_folder(folder_path, output_file)

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

class SymmetricCrossEntropyLoss(nn.Module):
    """
    SCE = α · CE + β · RCE  (Ghosh et al., 2021)
    • CE  = Cross-Entropy classica                        (robusta < α)
    • RCE = Reversed Cross-Entropy (penalizza outlier)    (robusta < β)

    Args
    ----
    alpha        peso della CE         (tipico 0.1 – 1.0)
    beta         peso della RCE        (tipico   1  – 10 )
    num_classes  # classi del dataset
    epsilon      smoothing per evitare log(0) nel calcolo di RCE
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 num_classes: int = 6,
                 epsilon: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = epsilon
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()            # CE con reduction='mean'

    # ----------------------------------------------------------------- forward
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # --- CE classica -----------------------------------------------------
        ce_loss = self.ce(logits, targets)

        # --- RCE (reversed CE) ----------------------------------------------
        # p = modello (softmax);   ỹ = one-hot label (add smoothing)
        p = F.softmax(logits, dim=1)                               # [B, C]
        y_tilde = torch.zeros_like(p).scatter_(1, targets.unsqueeze(1), 1)
        y_tilde = y_tilde * (1 - self.eps) + self.eps / self.num_classes

        rce_loss = (- (p * torch.log(y_tilde)).sum(dim=1)).mean()

        # combinazione pesata
        return self.alpha * ce_loss + self.beta * rce_loss
    
class EarlyStop:
    def __init__(self, patience=10):
        self.patience  = patience
        self.best_val  = None
        self.counter   = 0

    def step(self, current_val):
        if self.best_val is None or current_val > self.best_val:
            self.best_val = current_val
            self.counter  = 0
            return False      # continua il training
        self.counter += 1
        return self.counter >= self.patience
