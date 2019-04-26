Repo provides scripts to train on multiple datasets of format of 6Brown or AMOS. The main script is HardNetMultipleDatasets.py

To run the code, you must first download the following datasets:

[AMOS_views_v3](https://drive.google.com/open?id=1Dza78UlrbHKG83XZlvNKOiAHVWg3uiHn) 
[6Brown](https://drive.google.com/drive/folders/1dxxsO8Ob2WLTHqa5nwRpsGhXfYwPSreV?usp=sharing) 
[HPatches view split](https://drive.google.com/file/d/1gQu4sQ7nZP-a2p_2ffRq0V3avEBlJhQz/view?usp=sharing_

into Datasets folder - see the definition in the source code. The script provides an easy way to define source datasets and the composition of patches (how many patches in each batch from each source). The default definition is the one we used to get weights for http://cvg.dsi.unifi.it/cvg/index.php?id=caip-2019-contest#results competition.

Example:
python -utt HardNetMultipleDatasets.py --id=1 --weight-function=Hessian --epochs=10 2>&1


