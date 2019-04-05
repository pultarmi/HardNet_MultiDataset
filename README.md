# HardNet_MultiDataset
Repo provides scripts to train on multiple datasets of format of 6Brown or AMOS. Use the script HardNetMultipleDatasets.py

Example:
python -utt HardNetMultipleDatasets.py --id=1 --tower-dataset=../Process_DS/Handpicked_v3_png --masks-dir=/mnt/home.stud/qqpultar/qqpultar/Process_DS/Cams_masks_all/png --weight-function=Hessian --training-set=delete --epochs=5 --dataroot=../Process_DS/Datasets_all 2>&1