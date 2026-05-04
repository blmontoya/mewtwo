import torch
from torch.utils.data import DataLoader
import hyperparameters as hp
import params as p

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    print(f"num of batches is: {len(loader)}")
    for batch_idx, (imgs, labels, _, _, _, _, _, _) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        logits = model(imgs)
        #breakpoint()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0: 
            print(f"loss for batch {batch_idx} is {loss}")

    return total_loss / len(loader)

def evaluate(model, loader, device, num_classes=p.NUM_CLASSES):
    model.eval()
    intersection = torch.zeros(num_classes)
    union = torch.zeros(num_classes)

    # Learned class index to name mapping from Semantic KITTI dataset
    class_names = {
        0: "unlabeled", 1: "car", 2: "bicycle", 3: "motorcycle", 4: "truck",
        5: "other-vehicle", 6: "person", 7: "bicyclist", 8: "motorcyclist",
        9: "road", 10: "parking", 11: "sidewalk", 12: "other-ground",
        13: "building", 14: "fence", 15: "vegetation", 16: "trunk",
        17: "terrain", 18: "pole", 19: "traffic-sign"
    }

    with torch.no_grad():
        for imgs, labels, mask, u, v, scan_name, seq, r in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            preds = model(imgs).argmax(dim=1)
            for cls in range(1, num_classes):
                pred_c = (preds == cls).cpu()
                label_c = (labels == cls).cpu()
                intersection[cls] += (pred_c & label_c).sum()
                union[cls] += (pred_c | label_c).sum()

    # IoU for each class except class 0 (since that is unlabeled)
    per_class_iou = {}
    for cls in range(1, num_classes):
        if union[cls] > 0:
            iou = (intersection[cls] / union[cls]).item()
            per_class_iou[class_names[cls]] = iou

    miou = sum(per_class_iou.values()) / len(per_class_iou)
    return miou, per_class_iou