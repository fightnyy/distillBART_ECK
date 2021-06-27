## Distill mbart For ECK

* Among mbart, I fine-tuned the models for English, Chinese, and Korean. 
* Each model has a different number of layers. (For example, if model denotes 12-3, it represent this model is composed of __12 Encoder__ and __3 Decoder__
* We plan to develop OpenSource called OpenSFT soon. Please look forward to it!



| __BLEU__ \ Model | 12-3 English | 12 -3 Korean | 12 - 3 Chinese |
| ---------------- | ------------ | ------------ | -------------- |
| __1st epoch__        | __53__           | __35__           | __27__             |
| 2nd epoch        | 52           | 35           | 25             |
| 3rd epoch        | 51           | 33           | 23             |

__Inference Time :__ 0.5S



| __BLEU__ \ Model | 9-3 English | 9 -3 Korean | 9 - 3 Chinese |
| ---------------- | ----------- | ----------- | ------------- |
| 1st epoch        | __54__      | __36__      | __24__        |
| 2nd epoch        |             |             |               |
| 3rd epoch        |             |             |               |
| 4th epoch        |             |             |               |

__Inference Time :__ 0.2S
