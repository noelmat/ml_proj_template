export DEVICE='cuda'
export TRAIN_FOLDS_CSV='../input/train_folds.csv'

export IMAGE_HEIGHT=137
export IMAGE_WIDTH=236


export TRAIN_BATCH_SIZE=64
export VALID_BATCH_SIZE=64
export EPOCHS=50

export MEAN='(0.485, 0.456, 0.406)'
export STD='(0.229, 0.224, 0.225)'


export BASE_MODEL='resnet34'

export TRAIN_FOLDS='(0,1,2,3)'
export VALID_FOLDS='(4,)'
python train.py

export TRAIN_FOLDS='(0,1,2,4)'
export VALID_FOLDS='(3,)'
python train.py

export TRAIN_FOLDS='(0,1,4,3)'
export VALID_FOLDS='(3,)'
python train.py

export TRAIN_FOLDS='(0,4,2,3)'
export VALID_FOLDS='(1,)'
python train.py

export TRAIN_FOLDS='(4,1,2,3)'
export VALID_FOLDS='(0,)'
python train.py
