# supplementary-physical-color-calibration
Supplementary code for Xiaoyi Ji et al., "Physical Color Calibration of Digital Pathology Scanners for Robust Artificial Intelligence Assisted Cancer Diagnosis". The study demonstrates that physical color calibration provides a potential solution to the variation introduced by different digital pathology scanners, making AI-based cancer diagnostics more reliable and applicable in diverse clinical settings. The full manuscript is available for reading [here](https://arxiv.org/abs/2307.05519). 

The purpose of this repository is to present the process of analyzing prostate biopsies in the study, detailing the steps of predictions for cancer presence, cancer length and Gleason grading with or without physical color calibration. Trained models and example data required for running the code in this repository can be downloaded from a separate record in Zenodo (see below).

If using the models or code in this repository for your research, please cite:

> Xiaoyi Ji, Richard Salmon, Nita Mulliqi, Umair Khan, Yinxi Wang, Anders Blilie, Henrik Olsson, Bodil Ginnerup Pedersen, Karina Dalsgaard Sørensen, Benedicte Parm Ulhøi, Svein R Kjosavik, Emilius AM Janssen, Mattias Rantalainen, Lars Egevad, Pekka Ruusuvuori, Martin Eklund, Kimmo Kartasalo, "Physical Color Calibration of Digital Pathology Scanners for Robust Artificial Intelligence Assisted Cancer Diagnosis". arXiv preprint arXiv:2307.05519, 2023.

 ---

## Files
As a first step after cloning the repository, download and unzip models.zip, example_data.zip and example_results.zip from [https://zenodo.org/doi/10.5281/zenodo.10532598](https://zenodo.org/doi/10.5281/zenodo.10532598).

- **example_data**: Folder with tiles (.jpg) from one example slide, with and without color calibration.
- **example_results**: Folder with tile- and slide-level predictions for the example slide, with and without color calibration.
- **models**: Folder with tile- and slide-level models trained in the study with and without color calibrated data.
- **profiles**: Folder with ICC color profiles for calibration.
   1. sRGB_v4_ICC_preference.icc: chosen standard target color profile, which can be downloaded from the [official website](https://www.color.org/srgbprofiles.xalter).
   2. (scanner-specific profiles)
- **docker**: Folder with files needed to build a Docker image for running predictions.
- **src**: Folder with scripts for running predictions.
   1. **run_prediction_tiles.py**: Script for running tile-level prediction with DNNs.
   2. **run_prediction_slides.py**: Script for running slide-level prediction with gradient boosted trees.
   3. **dnn**: Folder with modules for tile-level prediction with DNNs.
   4. **xgboost**: Folder with modules for slide-level prediction with gradient boosted trees.

## Applying color calibration
Provided that a scanner-specific ICC color profile and a target profile are available, physical color calibration can be applied in Python following the sample workflow below. Scanner-specific profiles can be obtained by profiling the Sierra slide. This example uses **example_icc_profile.icc** as the scanner's ICC profile, **sRGB_v4_ICC_preference.icc** as the target profile, and **example_tile.jpg** as an input image tile. The color calibration is performed using the *ImageCms* module (version 1.0.0) within the *Pillow* library (version 8.0.0).
```
from PIL import Image, ImageCms

# Read a non-calibrated input image.
tile = Image.open('example_tile.jpg')

# Apply transformation from source to target color profile.
tile_calibrated = ImageCms.profileToProfile(tile, 'example_icc_profile.icc', 'sRGB_v4_ICC_preference.icc')

# Write the calibrated output tile.
tile_calibrated.save('example_tile_calibrated.jpg', quality=80)
```
## Building docker image
A Docker image providing all the necessary software dependencies for running the code can be built using the Dockerfile in this repo:

`$ cd docker`

`$ docker build -t your_docker_name .`

Having built the Docker image, you can start an interactive shell running in the container, with this repository mounted for accessing the files:

`$ docker run -it --name <your_docker_name> --workdir=<path_to_this_folder> --mount type=bind,src=<path_to_this_folder>,dst=<path_to_this_folder> <your_docker_name> /bin/bash`

## Running predictions
The codes in this example are to be run inside the Docker container built in the previous step.
### Tile-level prediction
`python3 -m src.run_prediction_tiles`
 
This step generates tile-level predictions of the example slide using trained DNN models, saving results (.csv) for both cancer detection and grading. By default the example images without color calibration are used. To generate predictions for the calibrated example images, you can use:

`python3 -m src.run_prediction_tiles --path_in_tiles example_data/calibration/slide_1 --path_out_tilepredictions example_results/calibration/slide_1 --path_in_model_cancer models/calibration/tile_level/cancer --path_in_model_grading models/calibration/tile_level/grading`

The path for saving predictions can also be edited to your own folder by `--path_out_tilepredictions your/target/folder`. The model is an ensemble consisting of 10 DNNs for cancer detection and 10 DNNs for Gleason grading.

The table below provides a sample of the grading output at the tile level, showing the first two rows for illustration. Each column labeled **tile_pred_class_*** denotes the predicted probability for a respective class (benign or malignant for detection, benign or Gleason pattern 3, 4, 5 for grading).

| tile_name    | slide       | tile_pred_class_0 | tile_pred_class_1 | tile_pred_class_2| tile_pred_class_3 |
|--------------|-------------|-------------------|-------------------|------------------|-------------------| 
| example_data/original/slide_1/21172_22245_42612_43685.jpg| example_data/original/slide_1 | 0.58203125 | 0.06088257 | 0.3569336 | 0.00003629923     |  
| example_data/original/slide_1/7772_8845_8844_9917.jpg | example_data/original/slide_1 |0.00000023841858| 0.00017130375 | 0.9995117 | 0.00033855438|

### Slide-level prediction
`python3 -m src.run_prediction_slides`

This step uses tile-level predictions from the DNNs and the trained slide-level models to run predictions on slide-level. By default the example images without color calibration are used. To generate predictions for the calibrated example images, you can use:

`python3 -m src.run_prediction_slides --path_in_tilepredictions example_results/calibration/slide_1 --path_out_slidepredictions example_results/calibration/slide_1/predictions_slides.csv --path_in_model_cancer models/calibration/slide_level/xgbmodel_cancer.pkl --path_in_model_grading models/calibration/slide_level/xgbmodel_grading.pkl --path_in_model_length models/calibration/slide_level/xgbmodel_length.pkl --threshold_cancer 0.423473`

Note that the `--path_in_tilepredictions` should be consistent with the `--path_out_tilepredictions` in the previous step. The path for saving slide-level predictions can also be edited in your own way by `--path_out_slidepredictions your/target/folder/predictions_slides.csv`.

The following table displays a sample prediction outcome for the example slide. Columns labeled **slide_pred_isup_*** represent the predicted probabilities for each International Society of Urological Pathology (ISUP) grade group (benign or ISUP grades 1 to 5). The most likely ISUP grade is indicated by **slide_class_isup**. Cancer detection prediction is also expressed as a probability, and cancer length prediction is measured in millimeters.

| slide | slide_pred_isup_0 | slide_pred_isup_1 | slide_pred_isup_2 | slide_pred_isup_3 | slide_pred_isup_4 | slide_pred_isup_5 | slide_class_isup | slide_pred_cancer | slide_pred_length |
|--------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|------------------|-------------------|-------------------|
| example_data/original/slide_1| 0.0000540614128112793 | 0.0452423321376226 | 0.0643287375237562 | 0.108125641384739   |  0.32057634253227    | 0.426987752766042 | 5 | 0.99994594| 7.287575|    

---

## Contact
For further questions or details, please reach out to Xiaoyi Ji ([xiaoyi.ji@ki.se](mailto:xiaoyi.ji@ki.se)) and Dr. Kimmo Kartasalo ([kimmo.kartasalo@ki.se](mailto:kimmo.kartasalo@ki.se)).

