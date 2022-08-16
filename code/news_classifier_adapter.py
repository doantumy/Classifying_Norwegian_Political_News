import os
import pandas as pd
import torch
from torch.utils import data
from transformers import AdamW
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
from random import random, randint
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed, EarlyStoppingCallback
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import BertConfig, BertModelWithHeads
import gc
import wandb
import time, datetime

gc.collect()
set_seed(456)
torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NewsDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

	def __len__(self):
		return len(self.labels)

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	acc = accuracy_score(labels, preds)
	print("\nAccuracy: ", acc)
	table = pd.DataFrame({'preds':preds,
						'labels':labels})
	table.to_csv("./{0}/{1}_{2}_{3}_{4}_{5}_{6}.tsv".format(outputdir,modelname,maxlength,learningrate,weightdecay,batchsize,language), sep="\t")
	return {
		'accuracy': acc,
	}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	arg = parser.add_argument
	arg("--modelpath", default='NbAiLab/nb-bert-base')
	arg("--modelname", default='nb-bert')
	arg("--language", default='nor')
	arg("--traindataset",default='./data/balanced_train.tsv')
	arg("--testdataset",default='./data/test.tsv')
	arg("--epoch", type=int, default=15)
	arg("--maxlength", type=int, default=512)
	arg("--learningrate", type=float, default=5e-5)
	arg("--weightdecay", type=float, default=1e-4)
	arg("--warmupstep", type=int, default=1000)
	arg("--batchsize", type=int, default=32)
	arg("--numlabels", type=int, default=7)
	arg("--accumsteps", type=int, default=1)
	arg("--getsubsetdata", type=bool, default=False)
	arg("--projectname", default='news-mbert-nor')

	args = parser.parse_args()
	modelpath = args.modelpath
	modelname = args.modelname
	language = args.language
	traindataset = args.traindataset
	testdataset = args.testdataset
	epoch = args.epoch
	maxlength = args.maxlength
	learningrate = args.learningrate
	weightdecay = args.weightdecay
	warmupstep = args.warmupstep
	batchsize = args.batchsize
	numlabels = args.numlabels
	accumsteps = args.accumsteps
	getsubsetdata = args.getsubsetdata
	projectname = args.projectname

	wandb.init(project=projectname, entity="pvi")
	outputdir = 'adapter-{0}-L{1}-LR{2}-W{3}-B{4}-AC{5}-{6}'.format(modelname, maxlength, learningrate, weightdecay, batchsize, accumsteps, language)

	# Load traindataset
	raw_train_data = pd.read_csv(traindataset, sep='\t')
	raw_test_data = pd.read_csv(testdataset, sep='\t')

	print("Train on all data")
	x_train, x_val, y_train, y_val = train_test_split(list(raw_train_data['text']), 
														list(raw_train_data['label']), 
														random_state=456, shuffle=True, 
														stratify=list(raw_train_data['label']), 
														train_size=.8, test_size=.2)
	x_test = list(raw_test_data['text'])
	y_test = list(raw_test_data['label'])

	
	tokenizer = AutoTokenizer.from_pretrained(modelpath)
	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=maxlength)
	val_encodings = tokenizer(x_val, truncation=True, padding=True, max_length=maxlength)
	test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=maxlength)

	train_dataset = NewsDataset(train_encodings, y_train)
	val_dataset = NewsDataset(val_encodings, y_val)
	test_dataset = NewsDataset(test_encodings, y_test)

	# Config BERT Model
	config = BertConfig.from_pretrained(
	    modelpath,
	    num_labels=numlabels,
	)
	model = BertModelWithHeads.from_pretrained(
	    modelpath,
	    config=config,
	)
	# Add a new adapter
	model.add_adapter('nor-news-classifier', config="pfeiffer", set_active=True)
	# Add a matching classification head
	model.add_classification_head(
	    'nor-news-classifier',
	    num_labels=numlabels,
	  )
	
	# Activate the adapter
	model.train_adapter(['nor-news-classifier'])
	
	training_args = TrainingArguments(
		evaluation_strategy='epoch',
		save_strategy='epoch',
		learning_rate=learningrate,
		optim='adamw_hf',
		per_device_train_batch_size=batchsize,
		per_device_eval_batch_size=batchsize,
		num_train_epochs=epoch,
		weight_decay=weightdecay,
		warmup_steps=warmupstep,
		load_best_model_at_end=True,
		metric_for_best_model='accuracy',
		output_dir=outputdir,
		logging_steps=100,
		# fp16=True,
		gradient_accumulation_steps=accumsteps,
		report_to='wandb',
	)
	print('Starting to train model.')
	trainer = AdapterTrainer(
		model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		tokenizer=tokenizer,
		compute_metrics=compute_metrics,
		callbacks=[EarlyStoppingCallback(2, 0.0)],
	)
	start = datetime.datetime.now()
	trainer.train()
	end = datetime.datetime.now()
	diff = (end - start)
	diff_seconds = int(diff.total_seconds())
	minute_seconds, seconds = divmod(diff_seconds, 60)
	hours, minutes = divmod(minute_seconds, 60)
	print("Training time: "f"{hours}h {minutes}m {seconds}s")
	
	print("Testing model on test set \n")
	trainer_test = AdapterTrainer(
	  model,
	  args=training_args,
	  train_dataset=train_dataset,
	  eval_dataset=test_dataset,
	  compute_metrics=compute_metrics,
	)
	trainer_test.evaluate()
	

