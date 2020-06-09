**Training HardNet8 on multiple datasets**

Provided code allows to train HardNet8 (described in the [thesis](http://cmp.felk.cvut.cz/~qqpultar/Download/Master_thesis.pdf)) on multiple datasets of format of Liberty or AMOS. Use the script ftrain.py for training.

Pretrained models are [available](http://cmp.felk.cvut.cz/~qqpultar/Weights/Weights-HardNet8.zip) in ".pt" and ".jitpt" formats. The latter can be loaded via torch.jit.load(name) without any additional code.

If you want to train the model, you must first download and unzip the [datasets](http://cmp.felk.cvut.cz/~qqpultar/Download/Datasets.zip). Please make sure to use the Liberty dataset from this file because it contains also source image IDs.

**Example:**

Run this to train the universal HardNet8-Univ descriptor on AMOS and Liberty datasets.
```
python -utt ftrain.py --arch=h8 --ds=v4+lib --loss=tripletMargin++
```

This command was used to train HardNet8-PT submitted to [CVPR IMW 2020](https://vision.uvic.ca/image-matching-challenge/).
```
python -utt ftrain.py --arch=h8E512 --ds=lib+colo+notre --bs=9000 --mpos=0.5 --fewcams
```

In both cases, to get the final model, run the code in Notebooks/add_pca.ipynb (you have to change the name of model) to create new checkpoint with added PCA compression.


**Please cite us if you use this code:**

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

[Working hard to know your neighbor's margins: Local descriptor learning loss](http://cmp.felk.cvut.cz/~radenfil/publications/Mishchuk-NIPS17.pdf)
```
@article{HardNet2017,
    author 	= {Anastasiya Mishchuk, Dmytro Mishkin, Filip Radenovic, Jiri Matas},
    title 	= "{Working hard to know your neighbor's margins: Local descriptor learning loss}",
    year 	= 2017}
(c) 2017 by Anastasiia Mishchuk, Dmytro Mishkin
```


