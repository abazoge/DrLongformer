import logging
import os
import math
import copy
import torch
from torch import nn
from torch.nn import functional as F
from itertools import chain
from dataclasses import dataclass, field
from transformers import CamembertForMaskedLM, CamembertTokenizerFast, CamembertModel, DataCollatorForLanguageModeling, Trainer
from transformers import LongformerForMaskedLM, LongformerTokenizerFast, LongformerModel, TextDataset
from transformers import TrainingArguments, HfArgumentParser
from transformers import LongformerSelfAttention, AutoTokenizer
from datasets import load_from_disk, load_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def copy_proj_layers(model):
	for i, layer in enumerate(model.longformer.encoder.layer):
		layer.attention.self.query_global = copy.deepcopy(layer.attention.self.query)
		layer.attention.self.key_global = copy.deepcopy(layer.attention.self.key)
		layer.attention.self.value_global = copy.deepcopy(layer.attention.self.value)
	return model

def pretrain(args, model, tokenizer, model_path, max_size):

	tokenized_datasets = load_from_disk("/path_data")

	train_dataset = tokenized_datasets['train']

	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

	trainer = Trainer(
		model=model,
		args=args,
		data_collator=data_collator,
		train_dataset=train_dataset,
	)
	
	trainer.train()
	trainer.save_model()


@dataclass
class ModelArgs:
	attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
	max_pos: int = field(default=4096, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))

training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
	'--output_dir', 'output_dir',
	'--warmup_steps', '100',
	'--learning_rate', '0.00005',
	'--weight_decay', '0.01',
	'--adam_epsilon', '1e-6',
	'--num_train_epochs', '1',
	'--logging_steps', '100',
	'--save_steps', '100',
	'--max_grad_norm', '5.0',
	'--per_device_train_batch_size', '8',
	'--gradient_accumulation_steps', '4',
	'--do_train',
	'--report_to', 'tensorboard',
])

# path to pretraining dataset
training_args.train_datapath = './nachos_full_doc.txt'

model_path = f'DrBERT-{model_args.max_pos}'
if not os.path.exists(model_path):
	os.makedirs(model_path)

logger.info(f'Loading the model from {model_path}')
tokenizer = CamembertTokenizerFast.from_pretrained(model_path)
model = LongformerForMaskedLM.from_pretrained(model_path)

logger.info(f'Pretraining DrBERT-{model_args.max_pos} ... ')

pretrain(training_args, model, tokenizer, model_path=training_args.output_dir, max_size=model_args.max_pos)

path_before_copy = f'DrBERT-{model_args.max_pos}-bef'
model.save_pretrained(path_before_copy)
tokenizer.save_pretrained(path_before_copy)

logger.info(f'Copying local projection layers into global projection layers ... ')
model = copy_proj_layers(model)
logger.info(f'Saving model to {model_path}')
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)