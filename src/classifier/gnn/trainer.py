import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class GNNTrainer:
    def __init__(self, model, train_set, val_set, action_names, adj,
                 lr=1e-3, weight_decay=1e-4, batch_size=32, epochs=50,
                 patience=10, device=None, class_weights=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.adj = adj.to(self.device)
        self.action_names = action_names
        self.epochs = epochs
        self.patience = patience

        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        if class_weights is None:
            class_weights = _inverse_freq_weights(train_set, len(action_names))
        weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

    def _run_epoch(self, loader, train):
        self.model.train(train)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        torch.set_grad_enabled(train)
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x, self.adj)
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
        return total_loss / max(1, total_samples), total_correct / max(1, total_samples)

    def fit(self, save_path=None):
        history = []
        best_macro_f1 = -1.0
        epochs_no_improve = 0

        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            report = self.evaluate()
            macro_f1 = sum(report[c]["f1"] for c in self.action_names) / len(self.action_names)

            history.append({
                "epoch": epoch,
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_loss, "val_acc": val_acc,
                "val_macro_f1": macro_f1,
            })

            print(f"Epoch {epoch:3d} | "
                  f"train {tr_loss:.4f}/{tr_acc:.3f} | "
                  f"val {val_loss:.4f}/{val_acc:.3f} | "
                  f"macroF1 {macro_f1:.3f}")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                epochs_no_improve = 0
                if save_path is not None:
                    torch.save(self.model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stop at epoch {epoch}. Best macroF1 {best_macro_f1:.3f}")
                    break

        return history, best_macro_f1

    def evaluate(self):
        self.model.eval()
        K = len(self.action_names)
        tp = [0] * K
        fp = [0] * K
        fn = [0] * K

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = torch.argmax(self.model(x, self.adj), dim=1)
                for c in range(K):
                    tp[c] += ((preds == c) & (y == c)).sum().item()
                    fp[c] += ((preds == c) & (y != c)).sum().item()
                    fn[c] += ((preds != c) & (y == c)).sum().item()

        report = {}
        for c, name in enumerate(self.action_names):
            p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] > 0 else 0.0
            r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] > 0 else 0.0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
            report[name] = {"precision": p, "recall": r, "f1": f1, "support": tp[c] + fn[c]}
        return report

    # Rows = true class, columns = predicted class
    def confusion_matrix(self):
        import numpy as np
        K = len(self.action_names)
        cm = np.zeros((K, K), dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = torch.argmax(self.model(x, self.adj), dim=1)
                for t, p in zip(y.cpu().numpy(), preds.cpu().numpy()):
                    cm[t, p] += 1
        return cm


def _inverse_freq_weights(dataset, num_classes):
    counts = [0] * num_classes
    for _, y in dataset.samples:
        counts[y] += 1
    total = sum(counts)
    # 1 / freq, then renormalize so weights average to 1
    weights = [total / (num_classes * max(1, c)) for c in counts]
    return weights
