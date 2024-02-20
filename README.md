# DrLongformer

<span style="font-size:larger;">**DrLongformer**</span> is a French pretrained Longformer model based on Clinical-Longformer that was further pre-trained on the NACHOS dataset (same dataset as [DrBERT](https://github.com/qanastek/DrBERT)). This model allows up to 4,096 tokens as input. DrLongformer consistently outperforms medical BERT-based models across most downstream tasks regardless of sequence length, except on NER tasks. Those downstream experiments broadly cover named entity recognition (NER), question answering (MCQA), Semantic textual similarity (STS) and text classification tasks (CLS). For more details, please refer to [our paper]().

### Model pre-training
We explored multiple strategies for the adaptation of Longformer models to the French medical domain:
- Further pretraining. 
- Converting a medical BERT model to the Longformer architecture.
- Pretraining from scratch.

All Pretraining scripts to reproduce the experiments are available in this repository.
For the `from scratch` and `further pretraining strategies, the training scripts are the same as [DrBERT](https://github.com/qanastek/DrBERT), only the bash scripts are different and available in this repository.

All models were trained on the [Jean Zay](http://www.idris.fr/jean-zay/) French HPC.

| Model name | Corpus | Pretraining strategy | Sequence Length | Model URL |
| :------:       | :---: |  :---: | :---: | :---: |
| `DrLongformer` | NACHOS 7 GB  | Further pretraining of Clinical-Longformer | 4096 | [HuggingFace](https://huggingface.co/abazoge/DrLongformer) |
| `DrBERT-4096` | NACHOS 7 GB  | Conversion of [DrBERT-7B](https://huggingface.co/Dr-BERT/DrBERT-7GB) to the Longformer architecture | 4096 | [HuggingFace](https://huggingface.co/abazoge/DrBERT-4096) |
| `DrLongformer-FS (from scratch)` | NACHOS 7 GB  | Pretraining from scratch | 4096 | Not available |


### Model Usage
You can use DrLongformer directly from [Hugging Face's Transformers](https://github.com/huggingface/transformers):
```python
# !pip install transformers
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("abazoge/DrLongformer")
model = AutoModelForMaskedLM.from_pretrained("abazoge/DrLongformer")
```

### Citation
```

```
