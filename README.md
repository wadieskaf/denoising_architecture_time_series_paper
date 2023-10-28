# Official Implementation of the paper "Denoising Architecture for Unsupervised Anomaly Detection in Time-Series"

This repository contains the official implementation of the paper "Denoising Architecture for Unsupervised Anomaly Detection in Time-Series" by Wadie Skaf and Tomáš Horváth.

Springer: https://link.springer.com/chapter/10.1007/978-3-031-15743-1_17
ArXiv preprint: https://arxiv.org/abs/2208.14337

## Requirements
Please check the `requirements.txt` file for the required packages.

## Usage

### Getting the data
the dataset used in this paper is the Yahoo S5 dataset, which can be requested and downloaded from [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70). The dataset should be placed in the `Datasets` folder.

### Defining the experiment(s) parameters

Define the seq_len and the architecture parameters in `build_experiments_file.py` and run it to generate the experiments file.

### Running the experiments

1. Check the `exps.json` file and make sure the experiments are defined as you wish.
2. Run `python experiments.py` to run the experiments.
3. The results will be stored in `experiments_results` folder. Please refer to the `experiments.py` file for more details.
4. In case the CSV files are messed due to storing the architectures as lists, you can use the `fix_csv_files.py` file to fix them.

## Citation
If you find this code useful, please cite our paper:
```
@InProceedings{skaf_2022_denoising,
author="Skaf, Wadie
and Horv{\'a}th, Tom{\'a}{\v{s}}",
editor="Chiusano, Silvia
and Cerquitelli, Tania
and Wrembel, Robert
and N{\o}rv{\aa}g, Kjetil
and Catania, Barbara
and Vargas-Solar, Genoveva
and Zumpano, Ester",
title="Denoising Architecture for Unsupervised Anomaly Detection in Time-Series",
booktitle="New Trends in Database and Information Systems",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="178--187",
abstract="Anomalies in time-series provide insights of critical scenarios across a range of industries, from banking and aerospace to information technology, security, and medicine. However, identifying anomalies in time-series data is particularly challenging due to the imprecise definition of anomalies, the frequent absence of labels, and the enormously complex temporal correlations present in such data. The LSTM Autoencoder is an Encoder-Decoder scheme for Anomaly Detection based on Long Short Term Memory Networks that learns to reconstruct time-series behavior and then uses reconstruction error to identify abnormalities. We introduce the Denoising Architecture as a complement to this LSTM Encoder-Decoder model and investigate its effect on real-world as well as artificially generated datasets. We demonstrate that the proposed architecture increases both the accuracy and the training speed, thereby, making the LSTM Autoencoder more efficient for unsupervised anomaly detection tasks.",
isbn="978-3-031-15743-1"
}
```

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

## Note
The last modification to this code was made on 28-12-2021, and it is not maintained anymore. So, it might be the case of having some issues with the latest versions of the packages used in this code. Please feel free to contact me in case you have any questions.

