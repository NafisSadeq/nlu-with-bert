# NLU

This is an example of natural language understanding on some benchmark datasets such as MultiWoz. Specifically, we are performing multilabel-classification to solve the intent detection and slot filling problem. Some code in this repository have been adapted from [ConvLab2](https://github.com/thu-coai/ConvLab-2.git)

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)

## Installation

Require python 3.6

Clone this repository:
```
git clone https://github.com/NafisSadeq/nlu-with-bert.git
```

Install via pip. We strongly recommend using virtual environment.
```
cd nlu-with-bert
pip install -e .
python -m spacy download en_core_web_sm
```

## Dataset

Currently experiments are supported on Multiwoz 2.1, Multiwoz 2.2 and Google SGD dataset. For running experiments on a particular dataset, we have to run preprocess.py available on that dataset folder. This will convert that dataset in a format which is suitable for our train and evaluation scripts. See below for detail usage. 

## Usage
For training NLU model on MultiWoz 2.1 dataset

#### Preprocess data

Run preprocessing code under `multiwoz21` directory by running commands below

```sh
$ cd multiwoz21
$ python preprocess.py all
```

output processed data will be on `multiwoz21/data/all_data/` directory.

#### Train a model

Run training code by running command below:

```sh
$ cd ..
$ python train.py --config_path multiwoz21/configs/multiwoz_all_context.json
```

The model will be saved under `output_dir` of config_file. Also, it will be zipped as `zipped_model_path` in config_file. You can set the parameters associated with experiments in the config json file.

#### Test a model

Run evaluation code by running command below

```sh
$ python test.py --config_path multiwoz21/configs/multiwoz_all_context.json
```

The result (`output.json`) will be saved under `output_dir` of config_file. 

#### Test different baselines

Run script as below:

```
bash script.sh ${DEVICE_ID} ${MULTIWOZ_ID} ${BACKBONE_NAME}
```

`MULTIWOZ_ID` is 21 or 22; `BACKBONE_NAME` is '' 'roberta_' 'finbert_' or 'todbert_'.


