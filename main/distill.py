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

teacher_config = AsianBartConfig.from_pretrained(
    "hyunwoongko/asian-bart-ecjk"
)

encoder_teacher_layers= [
    i for i in range(teacher_config.encoder_layers)
]

decoder_teacher_layers= [
    i for i in range(teacher_config.decoder_layers)
]



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
    distill_config = make_config(num_encoder, num_decoder)
    student_encoder_layer, student_decoder_layer = make_layer(num_encoder, num_decoder)
    student = AsianBartForConditionalGeneration(distill_config)

    model = make_student_model(student)
    return model


def make_student_model(student: nn.Module, except_encoder_layers, execept_decoder_layer) -> nn.Module:
    teacher_state_dict = teacher_model.state_dict()
    student_state_dict = OrderedDict()
    i = 0
    for k, v in teacher_state_dict.items():

        if "decoder.layers.11" in k:
            k = k[:21] + "2" + k[23:]
            student_state_dict[k] = v

        if check(k, except_encoder_layers, execept_decoder_layer):
            continue

        else:
            try :
                if k[21:22].isnumeric():
                    if k[22:23].isnumeric():
                        k = k[:21] + f"{i}" + k[23:]
                    else:
                        k = k[:21] + f"{i}" + k[22:]
                    i+=1
            except:
                continue
            k = k[:21] + "0" + k[22:]
            if "decoder.layers.7" in k:
                k = k[:21] + "1" + k[22:]
            student_state_dict[k] = v
    # print(student_state_dict)
    student.load_state_dict(student_state_dict)
    return student


def make_config(num_decoder: int, num_encoder: int) -> json:
    base_model_config = AsianBartConfig.from_pretrained("hyunwoongko/asian-bart-ecjk")
    base_model_config.encoder_layers = num_encoder
    base_model_config.decoder_layers = num_decoder
    distill_config = base_model_config

    return distill_config


def check(k: List[str], execept_encoder_layer: List[str],execept_decoder_layer: List[str]):
    for except_layer in execept_encoder_layer:
        if except_layer in k:
            return True

    for except_layer in execept_decoder_layer:
        if except_layer in k:
            return True
    return False


def make_layer(
        n_encoder_target: int, n_decoder_target: int, mode: str = "default"
) :
    en_change=False
    de_change=False
    enc_space_limit = 0
    dec_space_limit = 0
    if n_encoder_target > 6:
        en_change = True
        n_encoder_target = teacher_config.encoder_layers - n_encoder_target

    if n_decoder_target > 6:
        de_change = True
        n_decoder_target = teacher_config.decoder_layers - n_decoder_target

    if n_encoder_target != 0:
        enc_space_limit = teacher_config.encoder_layers // n_encoder_target
    if n_decoder_target != 0:
        dec_space_limit = teacher_config.decoder_layers // n_decoder_target

    tmp_encoder_distill_layers = []
    tmp_decoder_distill_layers = []

    if mode == "default":
        enc_cnd_layers = combinations(encoder_teacher_layers, n_encoder_target)
        dec_cnd_layers = combinations(decoder_teacher_layers, n_decoder_target)

        for layers in enc_cnd_layers:
            for idx in range(1, len(layers)):
                if abs(layers[idx - 1] - layers[idx]) < enc_space_limit:
                    break
            else:
                tmp_encoder_distill_layers.append(layers)
        tmp_encoder_distill_layers = tmp_encoder_distill_layers[0]

        for layers in dec_cnd_layers:
            for idx in range(1, len(layers)):
                if abs(layers[idx - 1] - layers[idx]) < dec_space_limit:
                    break
            else:
                tmp_decoder_distill_layers.append(layers)
        tmp_decoder_distill_layers = tmp_decoder_distill_layers[0]
    elif mode == "start":
        tmp_encoder_distill_layers = encoder_teacher_layers[:n_encoder_target]
        tmp_decoder_distill_layers = decoder_teacher_layers[:n_decoder_target]
    elif mode == "end":
        tmp_encoder_distill_layers = encoder_teacher_layers[teacher_config.encoder_layers - n_encoder_target:]
        tmp_decoder_distill_layers = decoder_teacher_layers[teacher_config.decoder_layers - n_decoder_target:]
    else:
        raise ValueError("mode must be one of start, end, or default.")


    encoder_distill_layers = tmp_encoder_distill_layers
    decoder_distill_layers = tmp_decoder_distill_layers
    # import pdb;pdb.set_trace()
    if mode == "default" and en_change:
        encoder_distill_layers=list(set(encoder_teacher_layers) - set(tmp_encoder_distill_layers))
    if mode == "default" and de_change:
        decoder_distill_layers=list(set(decoder_teacher_layers) - set(tmp_decoder_distill_layers))

    encoder_distill_layers = list(set(encoder_teacher_layers) - set(encoder_distill_layers))
    decoder_distill_layers = list(set(decoder_teacher_layers) - set(decoder_distill_layers))
    final_enc_list=[]
    final_dec_list=[]
    for encoder_layer in encoder_distill_layers:
        final_enc_list.append(f"encoder.layers.{encoder_layer}")

    for decoder_layer in decoder_distill_layers:
        final_dec_list.append(f"decoder.layers.{decoder_layer}")

    return final_enc_list, final_dec_list # Except 해야 할 layer 제공

if __name__ == "__main__":
    print("Start distill.py")
    encoder_distill_layers, decoder_distill_layers = make_layer(n_encoder_target=8, n_decoder_target=10, mode = "default")
    print(f"encoder_distill_layers : {encoder_distill_layers}")
    print(f"decoder_distill_layers : {decoder_distill_layers}")