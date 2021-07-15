from data_handler import *
from torch.utils.data import DataLoader
from transformers import AdamW, BertTokenizer
from utilities import *
import torch.nn as nn
from loss_functions.focal_loss import FocalLoss
from loss_functions.class_balanced_loss import CB_loss
from transformers_local_rep.src.transformers.models.bert.modeling_bert import BertForMixupSequenceClassification, BertForSequenceClassification

# got 81% val LM_acc with single-author 3, BS 16, LR 0.00003

# ---- Variables Setup ---- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRETRAINING_WEIGHTS = '/data/cvg/maurice/logs/PolitBERT/'
DATA_PATH = 'data_MA/'
SAMPLING = True
OVERSAMPLING = False
BATCH_SIZE = 16
LR = 2e-5
L2 = 0
N_EPOCHS = 50
AUGMENT_EDA = True
CLASSES_FOR_HEAVIER_AUG = []
DROPOUT = 0.1
FOCAL_LOSS = False
FOCAL_ALPHA = [1.25, 0.75]
FOCAL_GAMMA = 2
CLASSBALANCED_LOSS = False
CB_BETA = 0.9999
SINGLE_CLASS = False
AUTHOR = 6
MIX_TRANS = False
MIX_ALPHA = 0.4
MIX_EMBED = True
MIX_MANI = False

if not FOCAL_LOSS:
    FOCAL_ALPHA = []
    # if gamma is 0, then the focal loss is not applied bc of the formula
    FOCAL_GAMMA = 0.
if not CLASSBALANCED_LOSS:
    CB_BETA = 0.
if not SINGLE_CLASS:
    AUTHOR = None
if SAMPLING:
    if OVERSAMPLING:
        SAMPLING = 'over'
    else:
        SAMPLING = 'under'

if not MIX_TRANS and not MIX_EMBED and not MIX_MANI:
    MIX_ALPHA = 0

assert sum(1 for mixer in [MIX_MANI, MIX_EMBED, MIX_TRANS] if mixer) < 2

MODEL_NAME = 'PolitBERT_MIXTRANS-{}_MIXEMB-{}_MIXMANI-{}_MIXALPHA-{}_AUTHOR-{}_SAMP-{}_LR-{}_BS-{}_L2-{}_EDA-{}_HEAVYAUG-{}_DO-{}_FL-{}_FA-{}_FG-{}_CB-{}_CBB-{}'.format(
    MIX_TRANS, MIX_EMBED, MIX_MANI, MIX_ALPHA, AUTHOR, SAMPLING, LR, BATCH_SIZE, L2, AUGMENT_EDA, CLASSES_FOR_HEAVIER_AUG, DROPOUT, FOCAL_LOSS, FOCAL_ALPHA,
    FOCAL_GAMMA, CLASSBALANCED_LOSS, CB_BETA)
print(MODEL_NAME)

# determine if the code is run on the local machine or cluster
if str(device) != 'cpu':
    LOG_PATH = '/data/cvg/maurice/logs/'
else:
    LOG_PATH = 'logs/'

MODEL_LOGS = LOG_PATH + MODEL_NAME + '/'

# create a folder for the project
if not os.path.exists(MODEL_LOGS):
    os.makedirs(MODEL_LOGS)
# setup the logger
logging.basicConfig(filename=MODEL_LOGS + 'modellog.log', level=logging.INFO)
logging.info(MODEL_NAME)

# ---- Pre-process Dataframe ---- #
# read the data used for pre-training so that we don't test on data we already trained on
train = pd.read_pickle(DATA_PATH + 'train_data_NoDup_4Words.pkl')
val = pd.read_pickle(DATA_PATH + 'val_data_NoDup_4Words.pkl')
test = pd.read_pickle(DATA_PATH + 'test_data_NoDup_4Words.pkl')

X_train = train['Sentence'].values.tolist()
y_train = train['Label'].values.tolist()
X_val = val['Sentence'].values.tolist()
y_val = val['Label'].values.tolist()
X_test = test['Sentence'].values.tolist()
y_test = test['Label'].values.tolist()

X = []
y = []

# add the val data to the training data
X.extend(X_train)
X.extend(X_val)
X_train = X

y.extend(y_train)
y.extend(y_val)
y_train = y

print(Counter(y_train))
# possibly augment the training data according to EDA
if AUGMENT_EDA:
    X_train, y_train = augment_dataset_with_EDA(X_train, y_train, CLASSES_FOR_HEAVIER_AUG)
print(Counter(y_train))

# adjust the labels if we have a single-class problem
# author label is then 1 and non-author label is 0
if SINGLE_CLASS:
    y_train = [1 if item == AUTHOR else 0 for item in y_train]
    y_test = [1 if item == AUTHOR else 0 for item in y_test]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
X_train = tokenizer(X_train, padding='max_length', truncation=True)
# X_val = tokenizer(X_val, padding='max_length', truncation=True)
X_test = tokenizer(X_test, padding='max_length', truncation=True)

# ---- Prepare Training ---- #
print("\n--- Preparing the training ----")

# create the datasets
# train_data = BertDatasetWithTokenizer(X_train, y_train, tokenizer, augment_data=True)
# val_data = BertDatasetWithTokenizer(X_val, y_val, tokenizer, augment_data=False)
# test_data = BertDatasetWithTokenizer(X_test, y_test, tokenizer, augment_data=False)
train_data = LM_Dataset(X_train, y_train)
# val_data = BertDataset(X_val, y_val)
test_data = LM_Dataset(X_test, y_test)

# create the sampler
if SAMPLING:
    train_weights = get_weights_for_each_sample(train_data, bert_dataset=True)
    train_sampler = get_sampler(train_weights, y_train, OVERSAMPLING)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=4, sampler=train_sampler)

else:
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)

# val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# create the model
if not MIX_TRANS and not MIX_EMBED and not MIX_MANI:
    model = BertForSequenceClassification.from_pretrained(PRETRAINING_WEIGHTS,
                                                          num_labels=len(set(y_train)),
                                                          output_attentions=False,
                                                          output_hidden_states=False,
                                                          hidden_dropout_prob=DROPOUT)
else:
    model = BertForMixupSequenceClassification.from_pretrained(PRETRAINING_WEIGHTS,
                                 num_labels=len(set(y_train)),
                                 output_attentions=False,
                                 output_hidden_states=False,
                                 hidden_dropout_prob=DROPOUT)

    model.mix_trans = MIX_TRANS
    model.mix_emb = MIX_EMBED
    model.mix_mani = MIX_MANI
    model.alpha = MIX_ALPHA



# enable paralization
model = nn.DataParallel(model)
model.to(device)

# load the fine-tuning PolitBERT

# create the optimizer
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=L2)

# create the loss function
if CLASSBALANCED_LOSS:
    samples_per_class = Counter(y_train)
    samples_sorted_by_key = sorted(samples_per_class.items())
    samples_sorted_by_key = [i[1] for i in samples_sorted_by_key]

    # gamma = 0 is equal to a "normal" balanced CE loss, gamma > 0 is equal to focal loss combined with CB
    loss_fn = CB_loss(samples_sorted_by_key, len(samples_sorted_by_key), CB_BETA, FOCAL_GAMMA)

elif FOCAL_LOSS:
    loss_fn = FocalLoss(torch.tensor(FOCAL_ALPHA), FOCAL_GAMMA)
else:
    loss_fn = None

# ---- Training ---- #
print("\n--- Starting the training ----")

train_LM(epochs=N_EPOCHS,
         model=model,
         train_loader=train_loader,
         test_loader=test_loader,
         optimizer=optimizer,
         general_logpath=LOG_PATH,
         model_logpath=MODEL_LOGS,
         modelname=MODEL_NAME,
         loss_fn=loss_fn,
         single_author=AUTHOR)

del model
del optimizer
