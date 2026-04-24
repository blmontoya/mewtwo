#* this is temporary just testing something out
from torch.utils.data import DataLoader
from projection import RangeData

def main():
    dataset = RangeData(root="dataset", sequences=["00", "01"])
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in loader: 
        (img, labels, mask) = batch
        print(img.shape)
        print(labels.shape)
        print(mask.shape)

if __name__ == '__main__':
    main()