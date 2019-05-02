**Training HardNet on multiple datasets (including AMOS views)**

Repo provides scripts to train on multiple datasets of format of 6Brown or AMOS. The main script is HardNetMultipleDatasets.py

To run the code, you must first download the datasets [AMOS Patches](http://cmp.felk.cvut.cz/~qqpultar/AMOS_views_v3.zip), 
[HPatches view split](http://cmp.felk.cvut.cz/~qqpultar/hpatches_split_view_train.pt) into Datasets folder - see the definition in the HardNetMultipleDatasets.py file. The script provides an easy way to define source datasets and the composition of batches (how many patches in each batch from each source). The default definition is the one we used to get weights for [WISW@CAIP2019](http://cvg.dsi.unifi.it/cvg/index.php?id=caip-2019-contest#results) competition.

HPatches view split pt file was generated via the script HPatchesDatasetCreator.py. If you want to create it yourself (you can generate illum split as well), you need to download [hpatches-release](https://github.com/hpatches/hpatches-dataset) folder, run the script and then you should get the exactly same file.

**Example:**
```
python -utt HardNetMultipleDatasets.py --id=1 --weight-function=Hessian --epochs=10 --name=example --batch-size=3072 2>&1
```

**Please cite us if you use this code:**

[Working hard to know your neighbor's margins: Local descriptor learning loss](http://cmp.felk.cvut.cz/~radenfil/publications/Mishchuk-NIPS17.pdf)
```
@article{HardNet2017,
    author 	= {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title 	= "{Working hard to know your neighbor's margins: Local descriptor learning loss}",
    year 	= 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin
```

[Leveraging Outdoor Webcams for Local Descriptor Learning](http://diglib.tugraz.at/download.php?id=5c5941d91cdd5&location=browse)
```
@article{HardNetAMOS2019,
    author 	= {Milan Pultar, Dmytro Mishkin, Jiri Matas},
    title  	= "{Leveraging Outdoor Webcams for Local Descriptor Learning}",
    year   	= 2019,
    month    	= feb,
    booktitle 	= {Proceedings of CVWW 2019}
}
```
