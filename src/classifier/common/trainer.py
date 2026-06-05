import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Generic trainer used by both the LSTM and the single-frame baseline
class BaseTrainer:
    def __init__(self, model, dataset, class_names,
                 lr=1e-3, batch_size=32, epochs=30,
                 device=None, val_split=0.2,
                 val_dataset=None):
        # Pass val_dataset to use a separate, pre-built val set. 
        # If val_dataset is None, falls back to a random
        # window-level split which can leak across train/val.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.class_names = class_names
        self.epochs = epochs

        if val_dataset is not None:
            self.train_set = dataset
            self.val_set = val_dataset
        else:
            val_size = max(1, int(len(dataset) * val_split))
            train_size = len(dataset) - val_size
            self.train_set, self.val_set = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_set, batch_size=batch_size, shuffle=False)

        # Loss + optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        torch.set_grad_enabled(train)
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

        torch.set_grad_enabled(True)

        avg_loss = total_loss / max(1, total_samples)
        acc = total_correct / max(1, total_samples)
        return avg_loss, acc

    def fit(self, save_path=None):
        best_val_acc = 0.0
        history = []

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc
            })

            print(f"Epoch {epoch}/{self.epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

            # Save best model
            if save_path is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model to {save_path}")

        return history

    # Per-class precision/recall/F1 over validation set
    def evaluate(self):
        self.model.eval()
        num_classes = len(self.class_names)

        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = torch.argmax(self.model(x), dim=1)

                for c in range(num_classes):
                    tp[c] += ((preds == c) & (y == c)).sum().item()
                    fp[c] += ((preds == c) & (y != c)).sum().item()
                    fn[c] += ((preds != c) & (y == c)).sum().item()

        report = {}
        for c, name in enumerate(self.class_names):
            precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
            recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            report[name] = {"precision": precision, "recall": recall, "f1": f1}

        return report

    # Rows = true class, columns = predicted class
    def confusion_matrix(self):
        import numpy as np
        K = len(self.class_names)
        cm = np.zeros((K, K), dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = torch.argmax(self.model(x), dim=1)
                for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                    cm[t, p] += 1
        return cm
