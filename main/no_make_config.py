from asian_bart import AsianBartForConditionalGeneration, AsianBartConfig
from itertools import combinations
from collections import OrderedDict
from typing import List

import torch.nn as nn
import json

"""
12_3

0,1,2,3,4,5,6,7,8,9,10,11 => 3, 7 ,12 
To Leverage All Knowledge
"""

teacher_model = AsianBartForConditionalGeneration.from_pretrained(
    "hyunwoongko/asian-bart-ecjk"
)
encoder_layer = []

decoder_layer_3 = [
    "decoder.layers.0",
    "decoder.layers.1",
    "decoder.layers.2",
    "decoder.layers.4",
    "decoder.layers.5",
    "decoder.layers.6",
    "decoder.layers.8",
    "decoder.layers.9",
    "decoder.layers.10",
]


def start(num_encoder: int, num_decoder: int) -> nn.Module:
    distill_12_3_config = make_config(num_decoder, num_encoder)

    student12_3 = AsianBartForConditionalGeneration(distill_12_3_config)
    model = make_student_model(student12_3)
    return model


def make_student_model(student12_3: nn.Module) -> nn.Module:
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = OrderedDict()
    for k, v in teacher_state_dict.items():
        if "decoder.layers.11" in k:
            k = k[:21] + "2" + k[23:]
            student_state_dict[k] = v

        if check(k, decoder_layer_3):
            continue

        else:
            if "decoder.layers.3" in k:
                k = k[:21] + "0" + k[22:]
            if "decoder.layers.7" in k:
                k = k[:21] + "1" + k[22:]
            student_state_dict[k] = v
    # print(student_state_dict)
    student12_3.load_state_dict(student_state_dict)
    return student12_3


def make_config(num_decoder: int, num_encoder: int) -> json:
    base_model_config = AsianBartConfig.from_pretrained("hyunwoongko/asian-bart-ecjk")
    base_model_config.encoder_layers = num_encoder
    base_model_config.decoder_layers = num_decoder
    distill12_3_config = base_model_config

    return distill12_3_config


def check(k: List[str], decoder_layer_3: List[str]):
    for except_layer in decoder_layer_3:
        if except_layer in k:
            return True
    return False


def make_layer(
        n_encoder_target: int, n_decoder_target: int, mode: str = "default"
) -> nn.Module:
    limit = len(teacher_model) // n_decoder_target

    if mode == "default":
        cnd_layers = combinations(teacher_model, n_decoder_target)
        final_cnd_list = []
        for layers in cnd_layers:
            for idx in range(1, len(layers)):
                if abs(layers[idx - 1] - layers[idx]) < limit:
                    break
            else:
                final_cnd_list.append(layers)
        student_layer = final_cnd_list[0]
    elif mode == "start":
        student_layer = teacher_model[:n_decoder_target]
    elif mode == "end":
        student_layer = teacher_model[len(teacher_model) - n_decoder_target:]
    else:
        raise ValueError("mode must be one of start, end, or default.")
    return student_layer
