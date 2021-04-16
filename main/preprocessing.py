from typing import Tuple
from torch.utils.data import DataLoader, WeightedRandomSampler
from pdb import set_trace
import numpy as np
import math
import torch

paws_train_set = []
paws_val_set = []
paws_test_set = []
def load_multilingual_dataset(
    dataset_path: str, batch_size: int, max_length: int = -1 , mode:str = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    load dataset and create DataLoader
    Args:
        dataset_path (str): dataset path to load
        batch_size (int): batch size
        max_length (int): max character size for input strings
            (if you inputted negative number, it means infinity)
        mode (str) : for test_result
    """
    if max_length < 0:
        max_length = math.inf
    train_src_set = (
        open("{}/train.src".format(dataset_path), "r", encoding="utf8")
        .read()
        .splitlines()
    )
    train_lang_set = (
        open("{}/train.lang".format(dataset_path), "r", encoding="utf8")
        .read()
        .splitlines()
    )
    train_tgt_set = (
        open("{}/train.tgt".format(dataset_path), "r", encoding="utf8")
        .read()
        .splitlines()
    )

    # train_set = [
    #     [src, tgt, lang]
    #     for src, tgt, lang in zip(train_src_set, train_tgt_set, train_lang_set)
    #     if len(src) < max_length
    #     and len(tgt) < max_length
    #     and len(src.replace(" ", "")) != 0
    #     and len(tgt.replace(" ", "")) != 0
    #     and lang[0:5] == lang[6:]
    # ]
    train_set=[]
    ko = 0
    en = 0
    zh = 0
    total = 0
    for src, tgt, lang in zip(train_src_set, train_tgt_set, train_lang_set):
        if len(src) < max_length and len(tgt) < max_length and len(src.replace(" ", "")) != 0 and len(tgt.replace(" ", "")) != 0 and lang[0:5] == lang[6:]:
            if lang == "ko_KR":
                ko +=1
            elif lang == "en_XX":
                en += 1
            elif lang == "zh_CN":
                zh += 1
            total+=1
            train_set.append([src,tgt,lang])


    # ko = ko / total
    # en = en / total
    # zh = zh / total


    ko=get_pawsx_data(dataset_path,"ko", "train",ko)
    en=get_pawsx_data(dataset_path,"en", "train",en)
    zh=get_pawsx_data(dataset_path,"zh", "train",zh)

    lang_sample_count = np.array([ko, en, zh])
    weight = 1. / lang_sample_count

    train_weight = []
    for _, _, lang in train_set:
        if lang == "ko_KR":
            train_weight.append(weight[0])
        elif lang == "en_XX":
            train_weight.append(weight[1])
        else :
            train_weight.append(weight[2])
    train_weight = np.array(train_weight)
    train_weight = torch.from_numpy(train_weight)
    train_weight = train_weight.double()

    sampler = WeightedRandomSampler(train_weight, len(train_weight))


    get_pawsx_data(dataset_path,"ko", "validation")
    get_pawsx_data(dataset_path,"en", "validation")
    get_pawsx_data(dataset_path,"zh", "validation")

    train_set.extend(paws_train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler = sampler

    )

    val_loader = DataLoader(
        paws_val_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )




    return train_loader, val_loader


def get_test_data(dataset_path):
    get_pawsx_data(dataset_path,"ko", "test")
    get_pawsx_data(dataset_path,"en", "test")
    get_pawsx_data(dataset_path,"zh", "test")


    return paws_test_set



def get_pawsx_data(dataset_path : str,lang : str, mode: str, count = None):
    """
    load paws_x dataset and create DataLoader
    Args:
        lang (str): language that you want to make dataset Ex) ko -> Korean, en-> English, zh -> chinese
        mode (int): the mode that you want to choose Ex) train, validation
    """
    for val in (open(f"{dataset_path}/{lang}/{mode}.tsv", "r", encoding="utf8").read().splitlines()):
        if len(val.split('\t'))!=4:
            continue
        _, s1, s2, label = val.split('\t')
        if label == 0 or s1 == 'sentence1':
            continue
        if lang == 'ko':
            lang_code = 'ko_KR'
        elif lang == 'en':
            lang_code = 'en_XX'
        elif lang == 'zh':
            lang_code = 'zh_CN'
        else:
            raise ValueError('parameter lang only allow "ko", "zh","en"')

        if mode == 'train':
            paws_train_set.append([s1, s2, lang_code])
        elif mode == "validation":
            paws_val_set.append([s1, s2, lang_code])
        elif mode == "test":
            paws_test_set.append([s1, s2, lang_code])
        else:
            raise ValueError("parameter mode only allow \"test\", \"validation\", \"test\"")
        if count is not None:
            count += 1

    if count is not None:
        return count



