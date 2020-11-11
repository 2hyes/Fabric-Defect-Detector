import torch
import os
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

""" 
step1. train : val : test split
- nodefect(정상 데이터) - 6 : 2 : 2 ( 85, 28, 28장 )
- defect(결함 데이터) - 0 : 1 : 1 ( 0, 52, 53장 )
"""

## data load
# noDefect
transform = transforms.Compose([transforms.ToTensor()])

nodefect_dataset = ImageFolder(root= "./data/NoDefect/", transform=transform)
nodefect_loader = torch.utils.data.DataLoader(nodefect_dataset)

# train : val : test = 6 : 2 : 2 로 split
train_size = int(0.6 * len(nodefect_dataset) +1) # 85
val_size = int(0.2 * len(nodefect_dataset)) # 28
test_size = len(nodefect_dataset) - train_size - val_size # 28
print(train_size, val_size, test_size)

train_dataset_split, val_dataset_split, test_dataset_split = torch.utils.data.random_split(nodefect_dataset, [train_size, val_size, test_size])

# defect : 0094_027_05.png 이미지 사이즈 오류 -> 삭제했음
defect_dataset = ImageFolder(root= "./data/Defect/", transform=transform)
defect_loader = torch.utils.data.DataLoader(defect_dataset)
# train : val : test = 0 : 1 : 1 로 split
val_size = int(0.5 * len(defect_dataset)) # 52
test_size = len(defect_dataset) - val_size # 53
print( val_size, test_size)

val_dataset_split2, test_dataset_split2 = torch.utils.data.random_split(defect_dataset, [val_size, test_size])

## split
train_split_loader = torch.utils.data.DataLoader(train_dataset_split)
val_nodefectsplit_loader = torch.utils.data.DataLoader(val_dataset_split)
val_defectsplit_loader = torch.utils.data.DataLoader(val_dataset_split2)
test_nodefectsplit_loader = torch.utils.data.DataLoader(test_dataset_split)
test_defectsplit_loader = torch.utils.data.DataLoader(test_dataset_split2)

""" 
step2. divided into patches
- dataloader로 불러온 이미지를 **_nodefect(or defect).png 형태로 저장
"""
def save_img(img, idx, targets, path, defectness):  
    if defectness == True: # defect 이미지일때
        save_path = os.path.join(path,"{0:02d}_defect.png".format(idx+28))
        print("{0:02d}_defect.png".format(idx+28))
    else: # nodefect 이미지일때
        save_path = os.path.join(path,"{0:02d}_nodefect.png".format(idx))
        print("{0:02d}_nodefect.png".format(idx))


  save_image(img, save_path)   #자른 이미지 저장

def process(path, defectness, loader):
    for batch_idx, (inputs, targets) in enumerate(loader):
        save_img(inputs[0], batch_idx, targets, path,defectness) 

## save
process("./data/nocrop/train", False, train_split_loader)
process("./data/nocrop/test", False, test_nodefectsplit_loader)
process("./data/nocrop/test", True, test_defectsplit_loader)
process("./data/nocrop/val", False, val_nodefectsplit_loader)
process("./data/nocrop/val_for_training", False, val_nodefectsplit_loader)
process("./data/nocrop/val", True, val_defectsplit_loader)
