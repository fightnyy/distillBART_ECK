"""
python get_bleu_score.py --hyp hyp.txt \
--ref ref.txt \
--lang en
"""
import argparse
from pororo import Pororo
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import html


def get_hypotheses(lines, tokenizer):
    hypotheses = []
    for line in lines:
        tokens = tokenizer(line)
        hypotheses.append(tokens)
    return hypotheses


def get_list_of_references(lines, tokenizer):
    list_of_references = []
    for line in lines:
        tokens = tokenizer(line)
        list_of_references.append([tokens])
    return list_of_references


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp", type=str, required=True, help="system output file path"
    )
    parser.add_argument("--ref", type=str, required=True, help="reference file path")
    parser.add_argument(
        "--lang", type=str, required=True, default="en", help="en | ko | ja | zh"
    )
    args = parser.parse_args()

    hyp = args.hyp
    ref = args.ref
    lang = args.lang

    if lang in ("ko", "zh", "ja"):
        tokenizer = Pororo(task="tokenization", lang=lang)
    else:
        tokenizer = Pororo(task="tokenization", lang="en")

    hyp_lines = open(hyp, "r", encoding="utf8").read().strip().splitlines()[:1000]
    ref_lines = open(ref, "r", encoding="utf8").read().strip().splitlines()[:1000]

    assert len(hyp_lines) == len(ref_lines)

    list_of_hypotheses = get_hypotheses(hyp_lines, tokenizer)
    list_of_references = get_list_of_references(ref_lines, tokenizer)
    score = corpus_bleu(
        list_of_references,
        list_of_hypotheses,
        auto_reweigh=True,
        smoothing_function=SmoothingFunction().method3,
    )
    print("BLEU SCORE = %.2f" % score)
