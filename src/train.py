import os
import ast
from dispatch_model import MODELS
from dataset import BengaliDatasetTrain
import torch
import torch.nn as nn
from tqdm import tqdm

def loss_fn(outputs, targets):
    o1,o2,o3 = outputs
    t1,t2,t3 = targets
    l1 = nn.CrossEntropyLoss()(o1,t1)
    l2 = nn.CrossEntropyLoss()(o2,t2)
    l3 = nn.CrossEntropyLoss()(o3,t3)

    return (l1 + l2 + l3)/3

def train(dataset,dataloader, device, model, optimizer):
    model.train()
    for bi, d in tqdm(enumerate(dataloader),total=int(len(dataset)/dataloader.batch_size)):
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)

        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        outputs = model(image)
        loss = loss_fn(outputs,targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
def evaluate(dataset,dataloader, device, model):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(dataloader),total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        image = d['image']
        grapheme_root = d['grapheme_root']
        vowel_diacritic = d['vowel_diacritic']
        consonant_diacritic = d['consonant_diacritic']

        image = image.to(device, dtype=torch.float)
        grapheme_root = grapheme_root.to(device, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(device, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(device, dtype=torch.long)

        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)

        outputs = model(image)
        loss = loss_fn(outputs,targets)

        final_loss += loss
    return final_loss/counter
                  



def main():
    DEVICE = os.environ.get('DEVICE')
    TRAIN_FOLDS_CSV = os.environ.get("TRAIN_FOLDS_CSV")
    
    IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT"))
    IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH"))


    TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE"))
    VALID_BATCH_SIZE = int(os.environ.get("VALID_BATCH_SIZE"))
    EPOCHS = int(os.environ.get("EPOCHS"))

    TRAIN_FOLDS = ast.literal_eval(os.environ.get("TRAIN_FOLDS"))
    VALID_FOLDS = ast.literal_eval(os.environ.get("VALID_FOLDS"))

    MEAN = ast.literal_eval(os.environ.get("MEAN"))
    STD = ast.literal_eval(os.environ.get("STD"))

    BASE_MODEL = os.environ.get('MODEL')

    train_dataset = BengaliDatasetTrain(
        TRAIN_FOLDS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        MEAN,
        STD
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    valid_dataset = BengaliDatasetTrain(
        VALID_FOLDS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        MEAN,
        STD
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    lr= 1e-4
    model = MODELS[BASE_MODEL](pretrained=True).to(device=DEVICE)
    optimizer = torch.optim.Adam(model.parameters,lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',
                                                        patience=5,factor=0.3,verbose=True)

    for epoch in range(EPOCHS):
        train(train_dataset,train_data_loader,DEVICE,model,optimizer)
        val_score = evaluate(valid_dataset,valid_data_loader,DEVICE,model)
        scheduler.step(val_score)
        torch.save(model.state_dict(),f"{BASE_MODEL}_fold{VALID_FOLDS[0]}.bin")


if __name__ == "__main__":
    main()