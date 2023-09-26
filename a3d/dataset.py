"""
Data preparation for Speech Commands v0.02 
Code copy/adapted from the SpeechBrain framework.
"""

import os
import glob
import torch
import torchaudio
import string
import random
import shutil
import logging
import numpy as np
import pandas as pd
import speechbrain as sb
from speechbrain.utils.data_utils import download_file
from speechbrain.dataio.dataio import read_audio

np.random.seed(1234)
logger = logging.getLogger(__name__)

GSC_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# List of all the words (i.e. classes) within the GSC v2 dataset
all_words = [
    "yes","no","up","down","left","right","on","off","stop","go",
    "zero","one","two","three","four","five","six","seven","eight",
    "nine","bed","bird","cat","dog","happy","house","marvin","sheila",
    "tree","wow","backward","forward","follow","learn","visual"
]


def prep_speechcommands(
    data_folder,
    save_folder,
    words_wanted=[
        "yes",
        "no",
        "up",
        "down",
        "left",
        "right",
        "on",
        "off",
        "stop",
        "go",
    ],
    skip_prep=False,
):
    """
    Prepares the Google Speech Commands V2 dataset.
    Arguments
    ---------
    data_folder : str
        path to dataset. If not present, it will be downloaded here.
    save_folder: str
        folder where to store the data manifest files.
    skip_prep: bool
        If True, skip data preparation.
    Example
    -------
    >>> data_folder = '/path/to/GSC'
    >>> prep_speechcommands(data_folder)
    """

    if skip_prep:
        return

    # If the data folders do not exist, we need to extract the data
    data_folder = os.path.join(data_folder, "speech_commands_v0.02")
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder, exist_ok=True)

        # Check for zip file and download if it doesn't exist
        tar_location = os.path.join(data_folder, "speech_commands_v0.02.tar.gz")
        if not os.path.exists(tar_location):
            download_file(GSC_URL, tar_location, unpack=True)
        else:
            logger.info("Extracting speech_commands_v0.02.tar.gz...")
            shutil.unpack_archive(tar_location, data_folder)

    # Define the words that we do not want to identify
    unknown_words = list(np.setdiff1d(all_words, words_wanted))

    # Create the splits
    with open(os.path.join(data_folder, "validation_list.txt"), "r") as f:
        validation_list = f.read().splitlines()

    with open(os.path.join(data_folder, "testing_list.txt"), "r") as f:
        testing_list = f.read().splitlines()

    convert_to_tuple = lambda x: (x.split("/")[0], x)
    validation_df = pd.DataFrame(map(convert_to_tuple, validation_list), columns=['label', 'wav'])
    testing_df = pd.DataFrame(map(convert_to_tuple, testing_list), columns=['label', 'wav'])

    # Get name of all files in the training folder
    all_files = glob.glob(os.path.join(data_folder, "*", "*.wav"))
    all_files = ["/".join(x.split("/")[-2:]) for x in all_files]
    all_df = pd.DataFrame(map(convert_to_tuple, all_files), columns=['label', 'wav'])

    # Compute the differences to get the training set
    diff = np.setdiff1d(all_df["wav"], validation_df["wav"])
    diff = np.setdiff1d(diff, testing_df["wav"])
    training_df = all_df[all_df["wav"].isin(diff)]

    # Get the __unknown__ label
    training_df = generate_unknown_data(training_df, words_wanted, unknown_words)
    validation_df = generate_unknown_data(validation_df, words_wanted, unknown_words)
    testing_df = generate_unknown_data(testing_df, words_wanted, unknown_words)

    # Filter the wanted words
    words_wanted = words_wanted + ["_unknown_"]
    training_df = training_df[training_df["label"].isin(words_wanted)]
    validation_df = validation_df[validation_df["label"].isin(words_wanted)]
    testing_df = testing_df[testing_df["label"].isin(words_wanted)]

    # Get metadata
    training_df = insert_metadata(training_df)
    validation_df = insert_metadata(validation_df)
    testing_df = insert_metadata(testing_df)

    # Generate the ID column
    training_df["ID"] = training_df["wav"].copy()
    validation_df["ID"] = validation_df["wav"].copy()
    testing_df["ID"] = testing_df["wav"].copy()

    # Generate silence data
    training_df = generate_silence_data(training_df, data_folder, all_df, words_wanted)
    validation_df = generate_silence_data(validation_df, data_folder, all_df, words_wanted)
    testing_df = generate_silence_data(testing_df, data_folder, all_df, words_wanted)

    # Concatenate the data_folder variable back to path
    training_df["wav"] = training_df["wav"].apply(lambda x: os.path.join(data_folder, x))
    validation_df["wav"] = validation_df["wav"].apply(lambda x: os.path.join(data_folder, x))
    testing_df["wav"] = testing_df["wav"].apply(lambda x: os.path.join(data_folder, x))

    # Save the splits
    training_df.to_csv(os.path.join(save_folder, "train.csv"), index=False)
    validation_df.to_csv(os.path.join(save_folder, "validation.csv"), index=False)
    testing_df.to_csv(os.path.join(save_folder, "test.csv"), index=False)

    return None


def insert_metadata(df):
    with pd.option_context('mode.chained_assignment', None):
        df["duration"] = 1.0
        df["start"] = 0
        df["stop"] = 16000

    return df

def generate_unknown_data(df, words_wanted, unknown_words):
    # Get indexes to drop
    num_unknown_samples = len(df[df["label"].isin(unknown_words)])
    num_keep = len(df[df["label"].isin(words_wanted)])//len(words_wanted)
    idx_unknown = df[df["label"].isin(unknown_words)].index
    idx_drop = np.random.choice(idx_unknown, num_unknown_samples - num_keep, replace=False)

    # Drop and replace labels
    with pd.option_context('mode.chained_assignment', None):
        df.drop(idx_drop, inplace=True)
        df.loc[df["label"].isin(unknown_words), "label"] = "_unknown_"

    return df


def generate_silence_data(df, data_folder, all_files, words_wanted):
    """Generates silence samples.
    Arguments
    ---------
    num_known_samples_per_split: int
        Total number of samples of known words for each split (i.e. set).
    splits: str
        Training, validation and test sets.
    data_folder: str
        path to dataset.
    percentage_silence: int
        How many silence samples to generate; relative to the total number of known words.
    """
    num_silence_samples = len(df[df["label"].isin(words_wanted)])//len(words_wanted)

    # Fetch all background noise wav files used to generate silence samples
    silence_paths = all_files[all_files["label"] == "_background_noise_"]["wav"].tolist()
    # Generate random silence samples
    # Assumes that the pytorch seed has been defined in the HyperPyYaml file
    num_silence_samples_per_path = int(
        num_silence_samples / len(silence_paths)
    )
    silence_list = []
    for silence_path in silence_paths:
        signal = read_audio(
            os.path.join(data_folder, silence_path)
        )
        random_starts = (
            (
                torch.rand(num_silence_samples_per_path)
                * (signal.shape[0] - 16001)
            )
            .type(torch.int)
            .tolist()
        )

        for i, random_start in enumerate(random_starts):
            _hash = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            silence_list.append({
                "ID": f"silence_{_hash}_{silence_path}",
                "wav": silence_path,
                "label": "silence",
                "duration": 1.0,
                "start": random_start,
                "stop": random_start + 16000
            })

    df = pd.concat([df, pd.DataFrame(silence_list)], axis=0)
    return df

def create_datasets_speechcommands(hparams):
    "Creates the datasets and their data processing pipelines."
    data_folder = hparams["data_folder"]

    # 1. Declarations:
    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["annotation_folder"], "train.csv"),
        replacements={"data_root": os.path.join(data_folder, 'speech_commands_v0.02')},
    ).filtered_sorted(sort_key="duration", reverse=False)

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["annotation_folder"], "validation.csv"),
        replacements={"data_root": os.path.join(data_folder, 'speech_commands_v0.02')},
    ).filtered_sorted(sort_key="duration", reverse=False)

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=os.path.join(hparams["annotation_folder"], "test.csv"),
        replacements={"data_root": os.path.join(data_folder, 'speech_commands_v0.02')},
    ).filtered_sorted(sort_key="duration", reverse=False)

    datasets = [train_data, valid_data, test_data]
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start, stop, duration):
        start = int(start)
        stop = int(stop)
        num_frames = stop - start
        sig, fs = torchaudio.load(
            wav, num_frames=num_frames, frame_offset=start
        )
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides("label", "target")
    def label_pipeline(label):
        yield label
        label_encoded = label_encoder.encode_sequence_torch([label])
        yield label_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Fit encoder:
    # Load or compute the label encoder (with multi-GPU DDP support)
    lab_enc_file = os.path.join(hparams["annotation_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file, from_didatasets=[train_data], output_key="label",
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "target"]
    )

    datasets = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }

    return datasets, label_encoder