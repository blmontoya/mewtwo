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
import yaml

def main():
    # a small test
    # dataset = RangeData(root="dataset", sequences=["00", "01"])
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # for batch in loader: 
    #     (img, labels, mask, u, v, _, _, proj_idx) = batch
    #     print(img.shape)
    #     print(labels.shape)
    #     print(mask.shape)

    # # ensure that the GPU is working 
    # print("CUDA available:", torch.cuda.is_available())
    # print("CUDA device count:", torch.cuda.device_count())
    # print("PyTorch CUDA version:", torch.version.cuda)

    # if torch.cuda.is_available():
    #     print("GPU:", torch.cuda.get_device_name(0))

    #breakpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #set which sequences we want as train, val, and test sets
    train = [1,2,3,4,5]
    val = [6,7]
    test = [2]

    splits = RangeData.get_splits(train, val, test)
    
    train_set = RangeData(p.ROOT, splits["train"])
    val_set   = RangeData(p.ROOT, splits["val"])

    train_loader = DataLoader(train_set, batch_size=hp.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=hp.BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = RangeNetCNN(in_channels=5, num_classes=p.NUM_CLASSES)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                              step_size=hp.STEP_SIZE, gamma=hp.GAMMA)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    #if want to start with pre-loaded weights model 
    # state_dict = torch.load('best_model.pt', weights_only=True)
    # model.load_state_dict(state_dict)
    
    best_miou = 0.0
    losses = []
    ious = []
    for epoch in range(hp.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        torch.save(model.state_dict(), "best_model.pt")

        # Prints out the mIoU and per-class IoU for the validation set at the end of each epoch
        miou, per_class_iou = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | mIoU: {miou:.4f}")
        for cls_name, iou in per_class_iou.items():
            print(f"  {cls_name:<20s}: {iou:.4f}")
        scheduler.step()

        # Saves the best miou checkpoint
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), "best_model.pt")
        
        losses.append(train_loss)
        ious.append(miou)
        if epoch % 10 == 0: 
            np.save("train_loss.npy", np.array(losses))
            np.save("mious.npy", np.array(ious))

    #reproject into 3D and create correct files for visualization
    # model.eval() 
    # for i in range(len(train_set)):
    #     input_img, _, _, u, v, scan_name, seq, proj_idx = train_set[i]
    #     input_img = input_img.unsqueeze(0).to(device) 
    #     with torch.no_grad():
    #         logits = model(input_img)
    #         pred_img = logits.argmax(dim=1)[0] #pick best class per pixel
    #         pred_img = pred_img.detach().cpu().numpy()

    #     u = torch.from_numpy(u).to(device)
    #     v = torch.from_numpy(v).to(device)

    #     #convert the logit labels back to the original classes
    #     with open("config/semantic-kitti.yaml", 'r') as f: 
    #         DATA = yaml.safe_load(f)
        
    #     lookup = DATA["learning_map_inv"]
    #     get_hash = np.vectorize(lookup.get, otypes=[int])
    #     label_img = get_hash(pred_img)
    #     label_img = torch.from_numpy(label_img).long().to(device)
        
    #     preds = label_img[v, u] #while we are still finishing up the postprocessing, use this
    #     #preds = postprocess(input_img, pred_img, u, v, proj_idx)

    #     #save to file set to use with Semantic KITTI API
    #     preds = preds.detach().cpu().numpy()
    #     save_dir = os.path.join("method_predictions", "sequences", seq, "predictions")
    #     os.makedirs(save_dir, exist_ok=True)
    #     filepath =  os.path.join(save_dir, f"{scan_name}.label")
    #     preds.tofile(filepath)

if __name__ == '__main__':
    main()