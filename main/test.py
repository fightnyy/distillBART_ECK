from main import DistillBart
from preprocessing import get_test_data
from asian_bart import AsianBartTokenizer
import os

"""
s1, s2, lang_code, lang_code
"""

if __name__ == "__main__":
    print("test_start")
    print("model_loading")
    check_list = [
        "paraphrase_mlbart_epoch=00-val_loss=0.14.ckpt",
        "paraphrase_mlbart_epoch=01-val_loss=0.25.ckpt",
        "paraphrase_mlbart_epoch=02-val_loss=0.34.ckpt",
        "paraphrase_mlbart_epoch=03-val_loss=0.29.ckpt",
    ]
    for i in range(4):
        path = check_list[i]
        model = (
            DistillBart()
            .load_from_checkpoint(
                checkpoint_path=f"drive/MyDrive/mlbart_ckpt/{check_list[i]}"
            )
            .model
        )
        print("model_loaded")

        print("tokenizer_loading")
        tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
        print("tokenizer_loaded")

        print("dataset_loading")
        dataset = get_test_data("drive/MyDrive/dataset/")
        print("dataset_loaded")

        src_list = []
        src_lang_list = []
        print(f"{i+1}'s turn")
        print(f"{check_list[i]}  file opend")
        f = open(f"{check_list[i]}gen{i}.txt", "w")
        label = open("label.txt", "w")
        print("*****************")
        print("Writing Start!")
        print("*****************")
        for s1, s2, lang_code in dataset:
            inputs = tokenizer.prepare_seq2seq_batch(src_texts=s1, src_langs=lang_code)
            gen_token = model.generate(
                **inputs, forced_bos_token_id=tokenizer.lang_code_to_id[lang_code]
            )
            # import pdb;pdb.set_trace()
            f.write(tokenizer.decode(gen_token[0][2:], skip_special_tokens=True) + "\n")
            label.write(s2 + "\n")
        print("*****************")
        print("Writing end!")
        print("*****************")
        f.close()
        label.close()
