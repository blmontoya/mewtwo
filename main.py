#* this is temporary just testing something out
import torch
from torch.utils.data import DataLoader
from projection import RangeData
from range_net_helpers import train_one_epoch, evaluate
from range_net import RangeNetCNN
import hyperparameters as hp
import params as p


def main():
    dataset = RangeData(root="dataset", sequences=["00", "01"])
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in loader: 
        (img, labels, mask) = batch
        print(img.shape)
        print(labels.shape)
        print(mask.shape)

    splits = RangeData.get_splits()
    
    train_set = RangeData(p.ROOT, splits["train"])
    val_set   = RangeData(p.ROOT, splits["val"])

    train_loader = DataLoader(train_set, batch_size=hp.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=hp.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = RangeNetCNN(in_channels=5, num_classes=20).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                              step_size=hp.STEP_SIZE, gamma=hp.GAMMA)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    best_miou = 0.0
    for epoch in range(hp.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        miou = evaluate(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | mIoU: {miou:.4f}")

        # Save best checkpoint
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "best_model.pt")

if __name__ == '__main__':
    main()