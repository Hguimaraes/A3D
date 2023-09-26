# A3D

Link to the full paper: [Click here]().

**tl;dr**: This is an explorative work to understand the impact of white-box adversarial attacks on systems constructed on top of self-supervised speech representations.

## Cite

```latex
```

## Abstract

> Self-supervised speech pre-training has emerged as a useful tool to extract representations from speech that can be used across different tasks. While these models are starting to appear in commercial systems, their robustness to so-called adversarial attacks have yet to be fully characterized. This paper evaluates the vulnerability of three self-supervised speech representations (wav2vec 2.0, HuBERT and WavLM) to three white-box adversarial attacks under different signal-to-noise ratios (SNR). The study uses keyword spotting as a downstream task and shows that the models are very vulnerable to attacks, even at high SNRs. The paper also investigates the transferability of attacks between models and analyses the generated noise patterns in order to develop more effective defence mechanisms. The modulation spectrum shows to be a potential tool for detection of adversarial attacks to speech systems.

## How to run

a3d was developed with a python-package structure. First, we need to clone the repository and install the package.
We recommend you to use a virtual environment for this.

```bash
$ git clone https://github.com/Hguimaraes/A3D
$ cd A3D
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -e .
```

### Data

The dataset used in our experiments is the [Google Speech Commands v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz).
Here, you can proceed two ways:

1. Skip this section and let the code download and extract the files for you
2. Download the files and place manually

For the second option, you must download the tar.gz file and place it inside a folder called *speech_commands_v0.02*. When you execute the training or evaluation pipelines, just point to the path where the *speech_commands_v0.02* folder is.

### Run

First, you need to train the clean models. For instance, to train the model based on WavLM, execute the following commands from the project root (A3D):

```bash
$ cd recipes/train_kws_models
$ python train.py hparams/config_wavlm_kws.yaml --data_folder /folder_to_extract_gsc --annotation_folder /folder_to_save_csv_annotations
```

You can change for the appropriate paths in your computer. This will generate the model checkpoints inside the folder *recipes/train_kws_models/logs/kws_baseline_wavlm*.

Next, you can execute the robustness evaluations in the folder *recipes/eval_robustness*. Here, you should take a moment to look into the configuration file from *hparams/config_kws.yaml*.

In special, if you are interested in running the transferability experiments, you may want to edit the following variables:

1. attacker_model_folder: path to the model used to generate the perturbations
2. victim_model_folder: path to the model to be attacked
3. attacker_encoder: encoder name
4. victim_encoder: encoder name

If you want to change the attack type, look for the variable **attack**.


Finally, to run from the project root (A3D):

```bash
$ cd recipes/eval_robustness
$ python eval_ks.py hparams/config_kws.yaml --data_folder /folder_to_extract_gsc --annotation_folder /folder_to_save_csv_annotations
```
