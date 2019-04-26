Repo provides scripts to train on multiple datasets of format of 6Brown or AMOS. The main script is HardNetMultipleDatasets.py

To run the code, you must first download the datasets [AMOS_views_v3](https://drive.google.com/open?id=1Dza78UlrbHKG83XZlvNKOiAHVWg3uiHn) 
[6Brown](https://drive.google.com/drive/folders/1dxxsO8Ob2WLTHqa5nwRpsGhXfYwPSreV?usp=sharing) 
[HPatches view split](https://drive.google.com/file/d/1gQu4sQ7nZP-a2p_2ffRq0V3avEBlJhQz/view?usp=sharing) into Datasets folder - see the definition in the HardNetMultipleDatasets.py file. The script provides an easy way to define source datasets and the composition of patches (how many patches in each batch from each source). The default definition is the one we used to get weights for http://cvg.dsi.unifi.it/cvg/index.php?id=caip-2019-contest#results competition.

Example:
```
python -utt HardNetMultipleDatasets.py --id=1 --weight-function=Hessian --epochs=10 2>&1
```

Please cite us if you use this code:
```
@article{HardNet2017,
    author = {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title = "{Working hard to know your neighbor's margins:Local descriptor learning loss}",
    year = 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin

@article{[HardNetAMOS2019](http://diglib.tugraz.at/download.php?id=5c5941d91cdd5&location=browse),
    author = {Milan Pultar, Dmytro Mishkin, Jiri Matas},
    title = "{Leveraging Outdoor Webcams for Local Descriptor Learning}",
    year = 2019,
    month = feb,
    booktitle = {Proceedings of CVWW 2019}
}
```
