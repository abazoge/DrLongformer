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

def create_long_model(
	longformer_model,
	longformer_tokenizer,
	bert_model,
	bert_tokenizer,
	save_model_to,
	attention_window,
	max_pos
):
	model_bert = CamembertForMaskedLM.from_pretrained(bert_model)
	tokenizer_bert = CamembertTokenizerFast.from_pretrained(bert_tokenizer, model_max_length=max_pos)
	
	model_longformer = LongformerForMaskedLM.from_pretrained(longformer_model)

	config = model_longformer.config

	config.bos_token_id = 5
	config.eos_token_id = 6
	config.sep_token_id = 6
	config.vocab_size = 32005
	config.layer_norm_eps = model_bert.config.layer_norm_eps

	# extend position embeddings
	tokenizer_bert.model_max_length = max_pos
	tokenizer_bert.init_kwargs['model_max_length'] = max_pos
	current_max_pos, embed_size = model_bert.roberta.embeddings.position_embeddings.weight.shape
	max_pos += 2  # NOTE: RoBERTa has positions 0,1 reserved, so embedding size is max position + 2
	assert max_pos > current_max_pos

	# modigy words embeddings
	model_longformer.longformer.embeddings.word_embeddings = model_bert.roberta.embeddings.word_embeddings

	# allocate a larger position embedding matrix
	new_pos_embed = model_longformer.longformer.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)

	# copy position embeddings over and over to initialize the new position embeddings
	new_pos_embed[0:2] = model_bert.roberta.embeddings.position_embeddings.weight[0:2]
	k = 2
	step = current_max_pos - 2
	while k < max_pos - 1:
		new_pos_embed[k:(k + step)] = model_bert.roberta.embeddings.position_embeddings.weight[2:]
		k += step
	model_longformer.longformer.embeddings.position_embeddings.weight.data = new_pos_embed
	
	# token type embeddings
	model_longformer.longformer.embeddings.token_type_embeddings = model_bert.roberta.embeddings.token_type_embeddings

	# layernorm
	model_longformer.longformer.embeddings.LayerNorm = model_bert.roberta.embeddings.LayerNorm

	# copy attention weight from DrBERT to Longformer
	for i, layer in enumerate(model_bert.roberta.encoder.layer):
		model_longformer.longformer.encoder.layer[i].attention.self.query = copy.deepcopy(layer.attention.self.query)
		model_longformer.longformer.encoder.layer[i].attention.self.key = copy.deepcopy(layer.attention.self.key)
		model_longformer.longformer.encoder.layer[i].attention.self.value = copy.deepcopy(layer.attention.self.value)

		model_longformer.longformer.encoder.layer[i].attention.self.query_global = copy.deepcopy(layer.attention.self.query)
		model_longformer.longformer.encoder.layer[i].attention.self.key_global = copy.deepcopy(layer.attention.self.key)
		model_longformer.longformer.encoder.layer[i].attention.self.value_global = copy.deepcopy(layer.attention.self.value)

		model_longformer.longformer.encoder.layer[i].attention.output.dense = copy.deepcopy(layer.attention.output.dense)
		model_longformer.longformer.encoder.layer[i].attention.output.LayerNorm = copy.deepcopy(layer.attention.output.LayerNorm)
		
		model_longformer.longformer.encoder.layer[i].intermediate.dense = copy.deepcopy(layer.intermediate.dense)
		
		model_longformer.longformer.encoder.layer[i].output.dense = copy.deepcopy(layer.output.dense)
		model_longformer.longformer.encoder.layer[i].output.LayerNorm = copy.deepcopy(layer.output.LayerNorm)


	# modify LM_head
	model_longformer.lm_head.dense = model_bert.lm_head.dense
	model_longformer.lm_head.layer_norm = model_bert.lm_head.layer_norm
	model_longformer.lm_head.decoder = model_bert.lm_head.decoder


	model_longformer.resize_token_embeddings(32005)
	model_longformer.tie_weights()

	logger.info(f'saving model to {save_model_to}')
	model_longformer.save_pretrained(save_model_to)
	tokenizer_bert.save_pretrained(save_model_to)
	return model_longformer, tokenizer_bert


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
	# '--max_steps', '400',
	'--logging_steps', '100',
	'--save_steps', '100',
	'--max_grad_norm', '5.0',
	'--per_device_train_batch_size', '8',  # 32GB gpu with fp32
	'--gradient_accumulation_steps', '4',
	'--do_train',
	'--report_to', 'tensorboard',
])

bert_name = '../../Ressources_NLP/Models/BERT/DrBERT_512_7GB_77740'
longformer_name = '../../Ressources_NLP/Models/Longformer/longformer-base-4096'

model_path = f'DrBERT-test-{model_args.max_pos}'
if not os.path.exists(model_path):
	os.makedirs(model_path)
	
logger.info(f'Converting DrBERT into DrBERT-{model_args.max_pos}')
model, tokenizer = create_long_model(
	save_model_to=model_path,
	longformer_model=longformer_name,
	longformer_tokenizer=longformer_name,
	bert_model=bert_name,
	bert_tokenizer=bert_name,
	attention_window=model_args.attention_window,
	max_pos=model_args.max_pos,
)

print(model)

logger.info(f'Saving model to {model_path}')
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)