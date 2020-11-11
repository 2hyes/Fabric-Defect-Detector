import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import glob

""" 
step2. divided into patches
- train, val_for_training(only nodefect) 이미지들을 패치로 쪼개서 tensor형태로 저장
"""

### 저장되어있는 모든 crop된 이미지들 불러오기(grayscale로) - images list에 저장
train_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob("./data/nocrop/train/*.png")]
val_for_training_images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob("./data/nocrop/val_for_training/*.png")]

""" 
patch로 쪼개주는 함수
: image list를 받아서, tensor로 변환 
-> .view(패치개수, 패치사이즈, 패치사이즈)
-> train_patches.shape = 85 256 64 64 / val_for_training_patches.shape = 28 256 64 64
"""
def patch(images):
    patch_size = 64
    for i in range(len(images)):
        images[i] = transforms.ToTensor()((images[i]))
        # unfold로 shape조절 --> view --> [patch개수, patch_size, patch_size]
        patches = images[i].data.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size) 
        patches = patches.contiguous().view(-1, patch_size, patch_size) 
        #print(patches.shape)
        if i == 0:
        patch = patches
        if i > 0:
        patches = torch.cat((patch, patches))
        patch = patches
    # print(patches.shape)

    patches = patches.view(len(images), -1, patch_size, patch_size) 
    return patches


"""
patch를 이미지로 gdrive에 저장해주는 함수
: patches텐서를 이미지번호 폴더에, 각 256개의 패치를 저장
"""
def save_patches(patches, set):
    import os
    patch_path="./data/nocrop/patch"
    if not os.path.isdir( patch_path ) :
        os.mkdir( patch_path )

    # cv2.imwrite사용하기 위해 patches를 numpy로 변환
    patches = patches.numpy()

    for i in range(1, patches.shape[0]+1): # original image 1 ~ 85번(폴더구분)
        for num in range(1, patches.shape[1]+1): # patch 정보는 파일이름에 저장
            path = os.path.join("./data/nocrop/patch", set, "image{0:02d}".format(i))
            if not os.path.isdir( path ) :
                os.mkdir( path )
            filename = "{0:03d}.png".format(num)
            print(path + "/" + filename)
            # image로 저장해서 다시 로드해오려고함
            # 이미지로저장하기위해 다시 *255
            cv2.imwrite(path + "/" + filename, patches[i-1,num-1,:]*255.0) 
    print("Saving pathces complete!")

train_patches = patch(train_images)
print(train_patches.shape) # 85 256 64 64 # 사이즈=64*64;이미지당 패치개수=256
val_for_training_patches = patch(val_for_training_images)
print(val_for_training_patches.shape) # 28 256 64 64
save_patches(train_patches, "train")
save_patches(val_for_training_patches, "val_for_training")