## Distill mbart For ECK

* Among mbart, I fine-tuned the models for English, Chinese, and Korean. 
* Each model has a different number of layers. (For example, if model denotes 12-3, it represent this model is composed of __12 Encoder__ and __3 Decoder__



| __BLEU__ \ Model | 12-3 English | 12 -3 Korean | 12 - 3 Chinese |
| ---------------- | ------------ | ------------ | -------------- |
| __1st epoch__        | 53           | 35           | 27             |
| 2nd epoch        | 52           | 35           | 25             |
| 3rd epoch        | 51           | 33           | 23             |

