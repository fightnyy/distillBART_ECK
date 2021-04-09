from .main import DistillBart
from .preprocessing import get_test_data
from asian_bart import AsianBartTokenizer
import os

"""
s1, s2, lang_code, lang_code
"""

if __name__ == "__main__":
    model = DistillBart(12,3).load_from_checkpoint(checkpoint_path="check/point/path").model
    tokenizer = AsianBartTokenizer.from_pretrained("hyunwoongko/asian-bart-ecjk")
    dataset = get_test_data(f'{os.getcwd()}/../dataset/',batch_size=4)
    src_list = []
    src_lang_list = []
    f = open("gen.txt", "w")
    label = open('label.txt','w')
    for s1,s2,lang_code,_ in dataset:
        inputs = tokenizer.prepare_seq2seq_batch(
            src_texts=[s1],
            src_langs=[lang_code]
        )
        gen_token=model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"])
        f.write(tokenizer.decode(gen_token, skip_special_tokens=True)[0]+"\n")
        label.write(s2+"\n")
    f.close()
    label.close()

