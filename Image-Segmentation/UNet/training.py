import torch
import torchvision
from dataprocessing import ImageDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = ImageDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ImageDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
# from utils import load_checkpoint,save_checkpoint, get_loaders, check_accuracy, save_predictions_as_imgs
LR = 1e-4
Device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size =16
num_epochs= 5
num_workers =2
image_height = 572
image_width= 572
pin_memory=True
train_img_dir=""
train_mask_dir=""
val_img_dir=""
val_mask_dir=""
LOAD_MODEL= False
def train(loader,model,optimizer,loss_fn,scaler):
    loop = tqdm(loader)
    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(Device)
        targets = targets.float().unsqueeze(1).to(Device)
    with torch.cuda.amp.autocast():
        predictions =  model(data)
        loss = loss_fn(predictions,targets)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose([A.Resize(height=image_height,width=image_width),A.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],max_pixel_value=255.0),ToTensorV2()])
    val_transforms = A.Compose([A.Resize(height=image_height,width=image_width), A.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0],max_pixel_value=255.0), ToTensorV2()])
    model = UNet().to(Device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(),lr=LR)
    train_loader, val_loader = get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transforms,
        num_workers,
        pin_memory
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=Device)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=Device)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=Device
        )
