# MemCon-POLYP

This repository contains the source code of MemCon-POLYP and baselines from the paper,
## Datasets
In this work, we conducted experiments using five Polyp datasets and two NeoPolyp datasets:
Kvasir : Dataset contains 1000 images with varying resolutions ranging from 720 × 576 to 1920 × 1072 pixels. Kvasir data is collected using endoscopic equipment at Vestre Viken Health Trust in Norway, annotated and verified by medical doctors (experienced endoscopists).

CVC-ClinicDB: Dataset is an openly available collection of 612 images extracted from 31 colonoscopy sequences, with a resolution of 384×288. The dataset is commonly utilized in medical image analysis, specifically in detecting polyps in colonoscopy videos through segmentation techniques.

CVC-ColonDB: Dataset is provided by the Machine Vision Group. These images were extracted from 15 brief colonoscopy videos and consist of 380 images of 574 × 500 pixels resolution.

CVC-T : Dataset is a subset of the larger Endoscene dataset and serves as a primary test set. It consists of 60 images extracted from 44 video sequences recorded from 36 patients.

ETIS-Larib : Dataset consists of 196 images with high resolution (1226 x 996).

BKAI-IGH NeoPolyp-Small and NeoPolyp-Large : The NeoPolyp-Small includes 1200 high-resolution images (1280 x 959) is separated into a training set of 1000 images and a test set of 200 images. The NeoPolyp-Large has 5277 images for training and 1353 for testing. Two datasets categorize polyps into neoplastic and non-neoplastic classes, indicated by red and green colors.

### Training and evaluating


Training and evaluating MemCon on Polyp datasets:
```
python3 train_polyp.py --config configs/polyp.yaml
python3 eval_polyp.py --config configs/polyp.yaml
```
Training and evaluating MemCon on Small-NeoPolyp datasets:
```
python3 train_neo_small.py --config configs/neo_small.yaml
python3 eval_neo_small.py --config configs/neo_small.yaml
```

Training and evaluating MemCon on Large-NeoPolyp datasets:
```
python3 train_neo_large.py --config configs/neo_large.yaml
python3 eval_neo_large.py --config configs/neo_large.yaml
```

Training with ReCo is expected to require 20GB of memory in a single GPU setting. 



### Other Notices
TODO
## Citation
TODO

## Contact
If you have any questions, please contact nguyenviethoai99@gmail.com.



