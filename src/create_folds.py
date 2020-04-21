import pandas as pd
from iterstrat import ml_stratifiers

if __name__ == "__main__":
    df = pd.read_csv("input/train.csv")
    print(df.head())
    df.loc[:,'kfold'] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = ml_stratifiers.MultilabelStratifiedKFold(n_splits=5)
    X = df.image_id.values
    y = df.loc[:,["grapheme_root","vowel_diacritic","consonant_diacritic"]].values

    for fold, (train_, val_) in enumerate(kf.split(X=X, y=y)):
        print(f"{train_}, {val_}")
        df.loc[val_,'kfold'] = fold

    print(df.kfold.value_counts())
    df.to_csv('input/train_folds.csv')