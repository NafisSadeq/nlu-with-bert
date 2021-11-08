# intuit-non-cui

This is an example of natural language understanding on MultiWoz dataset. Specifically, we are performing multilabel-classification to solve the intent detection and slot filling problem. This code is based on the implementation provided by [ConvLab2](https://github.com/thu-coai/ConvLab-2.git)

- [Installation](#installation)
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
$ python train.py --config_path multiwoz21/configs/bilstm_multiwoz_all.json
```

The model will be saved under `output_dir` of config_file. Also, it will be zipped as `zipped_model_path` in config_file. 

#### Test a model

Run evaluation code by running command below

```sh
$ python test.py --config_path multiwoz21/configs/bilstm_multiwoz_all.json
```

The result (`output.json`) will be saved under `output_dir` of config_file. 




