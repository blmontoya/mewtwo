#* this is temporary just testing something out
import torch
from torch.utils.data import DataLoader
import numpy as np
from projection import RangeData
from range_net_helpers import train_one_epoch, evaluate
from postprocess import postprocess
from range_net import RangeNetCNN
import hyperparameters as hp
import params as p
import os 


def main():
    dataset = RangeData(root="dataset", sequences=["00", "01"])
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for batch in loader: 
        (img, labels, mask, u, v, _, _, proj_idx) = batch
        # print(img.shape)
        # print(labels.shape)
        # print(mask.shape)

    #breakpoint()
    splits = RangeData.get_splits()
    
    train_set = RangeData(p.ROOT, splits["train"])
    val_set   = RangeData(p.ROOT, splits["val"])

    train_loader = DataLoader(train_set, batch_size=hp.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=hp.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = RangeNetCNN(in_channels=5, num_classes=20)
    model.to(device)
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

    #evaluate IoU via reprojection into 3D
    model.eval() 
    for i in range(len(dataset)):
        input_img, _, _, u, v, scan_name, seq, proj_idx = dataset[i]
        input_img = input_img.unsqueeze(0).to(device) 
        with torch.no_grad():
            logits = model(input_img)
            pred_img = logits.argmax(dim=1)[0] #pick best class per pixel
        
        u = torch.from_numpy(u).to(device)
        v = torch.from_numpy(v).to(device)

        preds = postprocess(input_img, pred_img, u, v, proj_idx)

        #save to file set to use with Semantic KITTI API
        preds = preds.cpu().numpy().astype(np.uint32) 
        save_dir = os.path.join("method_predictions", "sequences", seq, "predictions")
        os.makedirs(save_dir, exist_ok=True)
        filepath =  os.path.join(save_dir, f"{scan_name}.label")
        preds.tofile(filepath)

if __name__ == '__main__':
    main()