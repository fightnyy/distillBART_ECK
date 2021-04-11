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
    model = DistillBart(n_encoder= 12,n_decoder=3).load_from_checkpoint(checkpoint_path="drive/MyDrive/mlbart_ckpt/paraphrase_mlbart_epoch=00-val_loss=0.14.ckpt").model
    print("model_loaded")

    print("tokenizer_loading")
    tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
    print("tokenizer_loaded")

    print("dataset_loading")
    dataset = get_test_data(f'{os.getcwd()}/../dataset/',batch_size=4)
    print("dataset_loaded")

    src_list = []
    src_lang_list = []
    print("file opend")
    f = open("gen.txt", "w")
    label = open('label.txt','w')
    print("*****************")
    print("Writing Start!")
    print("*****************")
    for s1,s2,lang_code,_ in dataset:
        inputs = tokenizer.prepare_seq2seq_batch(
            src_texts=[s1],
            src_langs=[lang_code]
        )
        gen_token=model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])
        f.write(tokenizer.decode(gen_token, skip_special_tokens=True)[0]+"\n")
        label.write(s2+"\n")
    print("*****************")
    print("Writing end!")
    print("*****************")
    f.close()
    label.close()

