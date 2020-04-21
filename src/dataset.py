import pandas as pd
import numpy as np
from PIL import Image
import joblib
import torch
import albumentations

class BengaliDatasetTrain:
    def __init__(self, folds, image_height, image_width,mean,std):
        df = pd.read_csv("input/train_folds.csv")

        df = df[["image_id","grapheme_root","vowel_diacritic","consonant_diacritic","kfold"]]

        df = df[df.kfold.isin(folds)].reset_index(drop=True)
        self.image_ids = df.image_ids.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

        if len(folds)==1:
            self.aug = albumentations.Compose([
                albumentations.Resize(image_height,image_width,always_apply=True),
                albumentations.Normalize(mean,std,always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(image_height,image_width,always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5),
                albumentations.Normalize(mean,std,always_apply=True)                                                
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self,item):
        image = joblib.load(f"../input/image_pickles/{self.image_ids[item]}.pkl")
        image = image.reshape(137,236).astype(float)
        image = Image.fromarray(image).convert("RGB")
        image = self.aug(image=np.array(image))['image']
        image = image.transpose(image, (2,0,1)).astype(np.float32)

        return {
            'image' : torch.tensor(image, dtype=torch.float),
            'grapheme_root': torch.tensor(self.grapheme_root,dtype=torch.long),
            'vowel_diacritic': torch.tensor(self.vowel_diacritic,dtype=torch.long),
            'consonant_diacritic': torch.tensor(self.consonant_diacritic,dtype=torch.long)
        }
