from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset

import torch

paws_train_set = []
paws_val_set = []
paws_test_set = []


def load_multilingual_dataset(
    dataset_path: str, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    """
    load dataset and create DataLoader
    Args:
        dataset_path (str): dataset path to load
        batch_size (int): batch size
        max_length (int): max character size for input strings
            (if you inputted negative number, it means infinity)
        mode (str) : for test_result
    """

    get_pawsx_data(dataset_path, "ko", "train")
    get_pawsx_data(dataset_path, "en", "train")
    get_pawsx_data(dataset_path, "zh", "train")

    get_pawsx_data(dataset_path, "ko", "validation")
    get_pawsx_data(dataset_path, "en", "validation")
    get_pawsx_data(dataset_path, "zh", "validation")

    tensor_paws_train_set = TensorDataset(torch.tensor(paws_train_set))
    train_loader = DataLoader(
        tensor_paws_train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    tensor_paws_valid_set = TensorDataset(torch.tensor(paws_val_set))
    val_loader = DataLoader(
        tensor_paws_valid_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    return train_loader, val_loader


def get_test_data(dataset_path):
    get_pawsx_data(dataset_path, "ko", "test")
    get_pawsx_data(dataset_path, "en", "test")
    get_pawsx_data(dataset_path, "zh", "test")

    return paws_test_set


def get_pawsx_data(dataset_path: str, lang: str, mode: str):
    """
    load paws_x dataset and create DataLoader
    Args:
        lang (str): language that you want to make dataset Ex) ko -> Korean, en-> English, zh -> chinese
        mode (int): the mode that you want to choose Ex) train, validation
    """
    for val in (
        open(f"{dataset_path}/{lang}/{mode}.tsv", "r", encoding="utf8")
        .read()
        .splitlines()
    ):
        if len(val.split("\t")) != 4:
            continue
        _, s1, s2, label = val.split("\t")
        if label == 0 or s1 == "sentence1":
            continue
        if lang == "ko":
            lang_code = "ko_KR"
        elif lang == "en":
            lang_code = "en_XX"
        elif lang == "zh":
            lang_code = "zh_CN"
        else:
            raise ValueError('parameter lang only allow "ko", "zh","en"')

        if mode == "train":
            paws_train_set.append([s1, s2, lang_code])
        elif mode == "validation":
            paws_val_set.append([s1, s2, lang_code])
        elif mode == "test":
            paws_test_set.append([s1, s2, lang_code])
        else:
            raise ValueError('parameter mode only allow "test", "validation", "test"')
