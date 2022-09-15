# Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning

This repository contains code to train a self-supervised learning model on chest X-ray images that lack explicit annotations and evalute this model's performance on pathology-classification tasks.

[Fig 1.pdf](https://github.com/rajpurkarlab/CheXzero/files/9576904/Fig.1.pdf)

## Prerequisites
To clone all files:
```git clone https://github.com/rajpurkarlab/CheXzero.git```

To install Python dependencies:
```pip install -r requirements.txt```

## Data 
### Training Dataset
1. Navigate to [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) to download the training dataset. 
2. Copy the dataset into the `data/` directory.
3. Run `python preprocess_train_data.py`
4. This should preprocess the chest x-ray images into a hdf5 format used for training stored at `data/cxr.h5` and extract the impressions section as text from the corresponding chest x-ray radiology report stored at `data/mimic_impressions.csv` .

### Model Checkpoints
Model checkpoints of CheXzero pre-trained on MIMIC-CXR are publicly available at the following [link](https://drive.google.com/drive/folders/19YH2EALQTbkKXdJmKm3iaK8yPi9s1xc-?usp=sharing). Download files and save them in the `./models/` directory.

## Running Training
Run the following command to perform CheXzero pretraining. 
```bash
python run_train.py
```

### Arguments
* `--cxr_filepath` Directory to load chest x-ray image data from.
* `--txt_filepath` Directory to load radiology report impressions text from.

Use `-h` flag to see all optional arguments. 


