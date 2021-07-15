from transformers import AutoConfig, BertTokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling, BertForMaskedLM
from transformers import TrainingArguments, Trainer
from data_handler import preprocess_sentences_dataframe
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from sklearn.model_selection import train_test_split


# ---- Load the data ---- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'PolitBERT_Pretraining'
config = AutoConfig.from_pretrained('bert-base-uncased')


if str(device) != 'cpu':
    FINE_TUNE_DATAPATH = 'data_MA/All_unique_Sentences_18-10-2020.pkl'
    PRE_TRAIN_DATAPATH = 'data_MA/All_other_Sentences_01-11-2020.pkl'
    LOG_PATH = '/data/cvg/maurice/logs/'
else:
    FINE_TUNE_DATAPATH = 'data_MA/Sentences_Samples.pkl'
    PRE_TRAIN_DATAPATH = 'data_MA/Sentences_Samples.pkl'
    LOG_PATH = 'logs/'

MODEL_LOG = LOG_PATH + MODEL_NAME + '/'
DATA_PATH = MODEL_LOG + 'data/'
TB_PATH = MODEL_LOG + 'tb_logs/'

if not os.path.exists(MODEL_LOG):
    os.makedirs(MODEL_LOG)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)


# we don't want to pretrain on the test data since we need it to measure the finetuning performance
X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f, _, _, _ = preprocess_sentences_dataframe(FINE_TUNE_DATAPATH,
                                                                                                     multiclass=True,
                                                                                                     processing='normal',
                                                                                                     split_data=True,
                                                                                                     remove_trump_sentences=False,
                                                                                                     drop_short_sentences=False,
                                                                                                     remove_contractions=False,
                                                                                                     convert_to_lowercase=True,
                                                                                                     remove_stopwords=False)

# store the splitted data, since we need exactly this split afterwards for the actual training
pd.to_pickle(pd.DataFrame({'Sentence': X_test_f, 'Label': y_test_f}), DATA_PATH + 'test_data.pkl')
pd.to_pickle(pd.DataFrame({'Sentence': X_val_f, 'Label': y_val_f}), DATA_PATH + 'val_data.pkl')
pd.to_pickle(pd.DataFrame({'Sentence': X_train_f, 'Label': y_train_f}), DATA_PATH + 'train_data.pkl')


X = preprocess_sentences_dataframe(PRE_TRAIN_DATAPATH,
                                   multiclass=True,
                                   processing='normal',
                                   split_data=False,
                                   remove_trump_sentences=False,
                                   drop_short_sentences=False,
                                   remove_contractions=False,
                                   convert_to_lowercase=True,
                                   remove_stopwords=False)

# Add all other data for pretraining to one list
X_train_pre = []
X_train_pre.extend(X_train_f)
X_train_pre.extend(X_val_f)
X_train_pre.extend(X)

# take a small random subset as validation set of the pretraining
X_train_pre, X_val_pre = train_test_split(X_train_pre, shuffle=True, test_size=0.02)

# save the data as txt since the dataset-creator requires this format
with open(DATA_PATH + 'pre_training_trainset.txt', 'w') as f:
    for line in X_train_pre:
        f.write('%s\n' % line)
with open(DATA_PATH + 'pre_training_valset.txt', 'w') as f:
    for line in X_val_pre:
        f.write('%s\n' % line)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# create the datasets
train_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                      file_path=DATA_PATH + 'pre_training_trainset.txt',
                                      block_size=config.max_position_embeddings)

val_dataset = LineByLineTextDataset(tokenizer=tokenizer,
                                      file_path=DATA_PATH + 'pre_training_valset.txt',
                                      block_size=config.max_position_embeddings)

# create the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# load the initial BERT weights
model = BertForMaskedLM(config).from_pretrained('bert-base-uncased').to(device)
tb_writer = SummaryWriter(log_dir=TB_PATH)

# setup the training arguments
training_args = TrainingArguments(
    output_dir=MODEL_LOG,
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_eval=True,
    do_train=True,
    save_steps=50_000,
    eval_steps=25_000,
    save_total_limit=20,
    dataloader_num_workers=8,
    evaluate_during_training=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    prediction_loss_only=True,
    tb_writer=tb_writer)

# train the model
trainer.train()

# save the final model
trainer.save_model(MODEL_LOG)
