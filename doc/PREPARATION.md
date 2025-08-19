## Installation
Install pytorch>2.0
```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

Other requirements
```
pip install -r requirements.txt
```

## Data Preparation
Create `data` folder in `src`.

### SMPL
Place SMPL data in `src/data/SMPL-family`:
```
.
├── smpl
│   ├── J_regressor_body25.npy
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
```
SMPL pickle files are from [official website](https://smpl.is.tue.mpg.de/).

Body25 regressor can be downloaded [here](https://github.com/zju3dv/EasyMocap/blob/master/data/smplx/J_regressor_body25.npy).

### AMASS
Download all AMASS datasets from [here](https://amass.is.tue.mpg.de/). Place them in `src/data/AMASS`:
```
.
├── AMASS
│   ├── ACCAD
│   ├── BioMotionLab_NTroje
│   ├── ...
```