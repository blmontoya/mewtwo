import torch
from torch.utils.data import DataLoader
import hyperparameters as hp
import params as p

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for imgs, labels, _, _, _ in loader:
        imgs = imgs.cuda()
        labels = labels.cuda().long()

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, num_classes=20):
    model.eval()
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    with torch.no_grad():
        for imgs, labels, masks in loader:
            imgs = imgs.cuda()
            labels = labels.cuda().long()
            preds = model(imgs).argmax(dim=1)

            # Note to self: Class 0 is unlabeled and we skip it
            for cls in range(1, num_classes):
                pred_c = (preds == cls).cpu()
                label_c = (labels == cls).cpu()
                intersection[cls] += (pred_c & label_c).sum()
                union[cls] += (pred_c | label_c).sum()

    valid = union[1:] > 0
    iou = (intersection[1:][valid] / union[1:][valid])
    return iou.mean().item()