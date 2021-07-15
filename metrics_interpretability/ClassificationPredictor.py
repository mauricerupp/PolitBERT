from data_handler import *
from transformers import BertTokenizer, BertForSequenceClassification, XLNetTokenizer, XLNetForSequenceClassification, ElectraTokenizer, ElectraForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import softmax
import torch.nn as nn
from lime.lime_text import LimeTextExplainer


def predict_BERT_model(path_to_weights, path_to_data, LargeModel=False, multiclass=True, single_author=None):
    assert path_to_weights.endswith('.pt')
    if not multiclass:
        assert single_author is not None

    if LargeModel:
        model_kind = 'bert-large-uncased'
    else:
        model_kind = 'bert-base-uncased'

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    data = data[data['Sentence'].map(word_counter) > 3]
    data.reset_index(drop=True, inplace=True)
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    if not multiclass:
        labels = [1 if item == single_author else 0 for item in labels]

    # prepare the data
    tokenizer = BertTokenizer.from_pretrained(model_kind)
    X = tokenizer(sentences, truncation=True, padding=True)

    prediction_data = LM_Dataset(X, labels)

    data_loader = DataLoader(prediction_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare the model
    model = BertForSequenceClassification.from_pretrained(model_kind,
                                                          num_labels=7,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.to(device)
    model.load_state_dict(torch.load(path_to_weights))
    model.eval()

    # predict the batches and calculate the metrics
    all_pred_probs = []
    with torch.no_grad():
        all_labels = []
        all_preds = []
        for batch in data_loader:
            outputs, labels, all_preds, all_labels, pred_probs = process_predict_LM_batch(model, batch, all_preds, all_labels, return_probabilities=True)
            all_pred_probs.extend(pred_probs)

        matrix = confusion_matrix(all_labels, all_preds)
        acc_per_class = matrix.diagonal() / matrix.sum(1)
        f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')

        # get probability for positive class only
        if not multiclass:
            all_pred_probs = [p[1] for p in all_pred_probs]
        roc_score = roc_auc_score(y_true=all_labels, y_score=all_pred_probs, average='macro', multi_class='ovo')

    return matrix, acc_per_class, f1, roc_score


def explain_BERT_LIME(path_to_weights, sentence, LargeModel=False, multiclass=True, single_author=None):
    assert path_to_weights.endswith('.pt')
    if not multiclass:
        assert single_author is not None

    if LargeModel:
        model_kind = 'bert-large-uncased'
    else:
        model_kind = 'bert-base-uncased'

    # prepare the data
    tokenizer = BertTokenizer.from_pretrained(model_kind)


    # prepare the model
    model = BertForSequenceClassification.from_pretrained(model_kind,
                                                          num_labels=7,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.load_state_dict(torch.load(path_to_weights))

    model = nn.DataParallel(model, device_ids=[0,1])
    model.to(device)

    model.eval()

    def predict_s(sent):
        with torch.no_grad():
            X = tokenizer(sent, truncation=True, padding='max_length')
            input_ids = torch.tensor(X['input_ids']).unsqueeze(0).to(device)
            att_mask = torch.tensor(X['attention_mask']).unsqueeze(0).to(device)

            out = model(input_ids, attention_mask=att_mask)

            return softmax(out[0].detach().cpu().numpy(), axis=1)[0]


    explainer = LimeTextExplainer()

    exp = explainer.explain_instance(sentence, predict_s)
    #print(exp.available_labels())
    exp.save_to_file('/data/cvg/maurice/logs/lime/{}.html'.format(re.sub(' ', '', sentence)))




def predict_BERT_multi_ensemble(paths_to_weights, path_to_data, weighting, LargeModel=False):

    if LargeModel:
        model_kind = 'bert-large-uncased'
    else:
        model_kind = 'bert-base-uncased'

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    data = data[data['Sentence'].map(word_counter) > 3]
    data.reset_index(drop=True, inplace=True)
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    # prepare the data
    tokenizer = BertTokenizer.from_pretrained(model_kind)
    X = tokenizer(sentences, truncation=True, padding=True)

    prediction_data = LM_Dataset(X, labels)

    data_loader = DataLoader(prediction_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare the model
    all_pred_probs = []
    for path in paths_to_weights:
        model = BertForSequenceClassification.from_pretrained(model_kind,
                                                              num_labels=len(set(labels)),
                                                              output_attentions=False,
                                                              output_hidden_states=False)
        model.to(device)
        model.load_state_dict(torch.load(path))
        model.eval()

        with torch.no_grad():
            all_labels_of_model = []
            all_preds_of_model = []
            pred_probs_of_model = []
            for batch in data_loader:
                outputs, _, all_preds_of_model, all_labels_of_model, pred_probs = process_predict_LM_batch(model,
                                                                                                           batch,
                                                                                                           all_preds_of_model,
                                                                                                           all_labels_of_model,
                                                                                                           return_probabilities=True)
                pred_probs_of_model.extend(pred_probs)

            all_pred_probs.append(pred_probs_of_model)


        del model

    ensemble_probs = [weighting[0] * all_pred_probs[0][i] + weighting[1] * all_pred_probs[1][i] for i in range(len(all_pred_probs[1]))]

    all_preds = np.argmax(ensemble_probs, axis=1)

    matrix = confusion_matrix(labels, all_preds)
    acc_per_class = matrix.diagonal() / matrix.sum(1)
    f1 = f1_score(y_true=labels, y_pred=all_preds, average='macro')

    roc_score = roc_auc_score(y_true=labels, y_score=ensemble_probs, average='macro', multi_class='ovo')

    return matrix, acc_per_class, f1, roc_score


def predict_BERT_ensemble(ordered_model_paths, path_to_data, LargeModel=False):
    assert len(ordered_model_paths) == 7

    if LargeModel:
        model_kind = 'bert-large-uncased'
    else:
        model_kind = 'bert-base-uncased'

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    data = data[data['Sentence'].map(word_counter) > 3]
    data.reset_index(drop=True, inplace=True)
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    # prepare the data
    tokenizer = BertTokenizer.from_pretrained(model_kind)
    X = tokenizer(sentences, truncation=True, padding=True)

    prediction_data = LM_Dataset(X, [0] * len(labels))

    data_loader = DataLoader(prediction_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare models and get the prediction for each sample
    # all_pred_probs will have to form of all predictions of a model, then all predictions of the next model
    all_pred_probs = []
    for path in ordered_model_paths:
        model = BertForSequenceClassification.from_pretrained(model_kind,
                                                              num_labels=2,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
        model.to(device)
        model.load_state_dict(torch.load(path))
        model.eval()

        with torch.no_grad():
            all_labels_of_model = []
            all_preds_of_model = []
            for batch in data_loader:
                outputs, _, all_preds_of_model, all_labels_of_model, pred_probs = process_predict_LM_batch(model, batch, all_preds_of_model,
                                                                                                           all_labels_of_model,
                                                                                                           return_probabilities=True)

                # now I apply binary softmax and then decide which class it is, not the absolute values
                all_pred_probs.extend([p[1] for p in pred_probs])

        del model

    # reshape the list that each row are all the predictions from one model
    all_pred_probs = np.array(all_pred_probs).reshape((len(labels), 7), order='F')

    all_preds = np.argmax(all_pred_probs, axis=1)

    matrix = confusion_matrix(labels, all_preds)

    acc_per_class = matrix.diagonal() / matrix.sum(1)

    f1 = f1_score(y_true=labels, y_pred=all_preds, average='macro')
    roc_score = roc_auc_score(y_true=labels, y_score=softmax(all_pred_probs, axis=1), average='macro', multi_class='ovr')

    return matrix, acc_per_class, f1, roc_score


def predict_XLNet_model(path_to_weights, path_to_data, LargeModel=False):
    assert path_to_weights.endswith('.pt')

    if LargeModel:
        model_kind = 'xlnet-large-cased'
    else:
        model_kind = 'xlnet-base-cased'

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    data = data[data['Sentence'].map(word_counter) > 3]
    data.reset_index(drop=True, inplace=True)
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    # prepare the data
    tokenizer = XLNetTokenizer.from_pretrained(model_kind, do_lower_case=True)

    # tokenize the data
    X = tokenizer(sentences, return_tensors="pt", padding=True)
    prediction_data = LM_Dataset(X, labels)
    data_loader = DataLoader(prediction_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare the model
    model = XLNetForSequenceClassification.from_pretrained(model_kind,
                                                          num_labels=len(set(labels)))
    model.to(device)
    model.load_state_dict(torch.load(path_to_weights))
    model.eval()

    # predict the batches and calculate the metrics
    with torch.no_grad():
        all_labels = []
        all_preds = []
        all_pred_probs = []
        for batch in data_loader:
            outputs, labels, all_preds, all_labels, pred_probs = process_predict_LM_batch(model, batch, all_preds, all_labels, return_probabilities=True)
            all_pred_probs.extend(pred_probs)
        matrix = confusion_matrix(all_labels, all_preds)
        acc_per_class = matrix.diagonal() / matrix.sum(1)
        f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')
        roc_score = roc_auc_score(y_true=all_labels, y_score=all_pred_probs, average='macro', multi_class='ovr')

    return matrix, acc_per_class, f1, roc_score


def predict_ELECTRA_model(path_to_weights, path_to_data):
    assert path_to_weights.endswith('.pt')

    model_kind = 'google/electra-base-discriminator'

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    data = data[data['Sentence'].map(word_counter) > 3]
    data.reset_index(drop=True, inplace=True)
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    # prepare the data
    tokenizer = ElectraTokenizer.from_pretrained(model_kind, do_lower_case=True)

    # tokenize the data
    X = tokenizer(sentences, padding='max_length', truncation=True)
    prediction_data = LM_Dataset(X, labels)
    data_loader = DataLoader(prediction_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare the model
    model = ElectraForSequenceClassification.from_pretrained(model_kind,
                                                          num_labels=len(set(labels)))
    model.to(device)
    model.load_state_dict(torch.load(path_to_weights))
    model.eval()

    # predict the batches and calculate the metrics
    with torch.no_grad():
        all_labels = []
        all_preds = []
        all_pred_probs = []
        for batch in data_loader:
            outputs, labels, all_preds, all_labels, pred_probs = process_predict_LM_batch(model, batch, all_preds, all_labels, return_probabilities=True)
            all_pred_probs.extend(pred_probs)
        matrix = confusion_matrix(all_labels, all_preds)
        acc_per_class = matrix.diagonal() / matrix.sum(1)
        f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')
        roc_score = roc_auc_score(y_true=all_labels, y_score=all_pred_probs, average='macro', multi_class='ovr')

    return matrix, acc_per_class, f1, roc_score


def predict_NonLM_model(path_to_weights_and_voc, path_to_data, multiclass, single_author=None, sentence_length=300, packing=False):

    if not multiclass:
        assert single_author is not None
    # setup variables
    SENTENCE_LENGTH = sentence_length
    if packing:
        PRE_PAD = False
        PACKING = True
    else:
        PRE_PAD = True
        PACKING = False

    # create vocabulary
    voc = open(path_to_weights_and_voc + 'vocabulary.txt', 'r')
    voc = voc.read().splitlines()
    punct_remover = RegexpTokenizer(r'\w+')
    lookup_table = StaticTokenizerEncoder(voc[5:], min_occurrences=1, tokenize=lambda s: punct_remover.tokenize(s))

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    if not multiclass:
        labels = [1 if item == single_author else 0 for item in labels]


    # prepare data
    test_data = SentenceDataset(sentences, labels, lookup_table, SENTENCE_LENGTH, PRE_PAD, PACKING)
    dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare model
    model = torch.load(path_to_weights_and_voc + 'weights.pt')
    model.to(device)
    model.eval()

    with torch.no_grad():
        all_labels = []
        all_preds = []
        all_pred_probs = []
        for sample in dataloader:
            labels, output = process_predict_batch(model, multiclass, sample)

            if multiclass:
                all_pred_probs.extend(softmax(output.detach().cpu().numpy(), axis=1))
            else:
                all_pred_probs.extend((output.detach().cpu().numpy()))

            all_labels.extend(labels.detach().cpu().numpy())

            if multiclass:
                all_preds.extend(output.argmax(1).detach().cpu().numpy())
            else:
                all_preds.extend(torch.round(output).detach().cpu().numpy())

        matrix = confusion_matrix(all_labels, all_preds)
        acc_per_class = matrix.diagonal() / matrix.sum(1)
        f1 = f1_score(y_true=all_labels, y_pred=all_preds, average='macro')
        roc_score = roc_auc_score(y_true=all_labels, y_score=all_pred_probs, average='macro', multi_class='ovr')

    return matrix, acc_per_class, f1, roc_score


def predict_CNN_ensemble(ordered_model_paths, path_to_data):

    assert len(ordered_model_paths) == 7

    # setup variables
    SENTENCE_LENGTH = 400
    PRE_PAD = True
    PACKING = False

    # create vocabulary
    voc = open(ordered_model_paths[0] + 'vocabulary.txt', 'r')
    voc = voc.read().splitlines()
    punct_remover = RegexpTokenizer(r'\w+')
    lookup_table = StaticTokenizerEncoder(voc[5:], min_occurrences=1, tokenize=lambda s: punct_remover.tokenize(s))

    data = pd.read_pickle(path_to_data)

    # pre-process data as we did for each model
    data = data[data['Sentence'].map(word_counter) > 3]
    data.reset_index(drop=True, inplace=True)
    sentences = data['Sentence'].values.tolist()
    labels = data['Label'].values.tolist()

    # prepare data
    test_data = SentenceDataset(sentences, labels, lookup_table, SENTENCE_LENGTH, PRE_PAD, PACKING)
    dataloader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=4)

    # prepare models and get the prediction for each sample
    all_pred_probs = []
    for path in ordered_model_paths:
        model = torch.load(path + 'weights.pt')
        model.to(device)
        model.eval()

        with torch.no_grad():
            for sample in dataloader:
                _, output = process_predict_batch(model, False, sample)
                all_pred_probs.extend((output.detach().cpu().numpy()))


        del model

    # reshape the list that each row are all the predictions from one model
    all_pred_probs = np.array(all_pred_probs).reshape((len(labels),7), order='F')

    all_preds = np.argmax(all_pred_probs, axis=1)

    matrix = confusion_matrix(labels, all_preds)

    acc_per_class = matrix.diagonal() / matrix.sum(1)

    f1 = f1_score(y_true=labels, y_pred=all_preds, average='macro')
    roc_score = roc_auc_score(y_true=labels, y_score=softmax(all_pred_probs, axis=1), average='macro', multi_class='ovr')

    return matrix, acc_per_class, f1, roc_score


if __name__ == "__main__":

    explain_BERT_LIME(path_to_weights='/data/cvg/maurice/logs/0_Network_Backups/Final_Models/Multiclass/PolitBERT_Finetuning_cleaned_inclVAL_LR-2e-05_BS-16_L2-0_EDA-True/model-epoch-6.pt',
                      sentence='I made promises when I was a senator that I\'d help.')

    exit()


    m, a, f1, r = predict_BERT_model(path_to_data='/data/cvg/maurice/data_MA/split_data/test_data_NoDup_4Words.pkl',
                                     path_to_weights='/data/cvg/maurice/logs/PolitBERT_MIXTRANS-False_MIXEMB-True_MIXMANI-False_MIXALPHA-0.4_AUTHOR-None_SAMP-under_LR-2e-05_BS-16_L2-0_EDA-True_HEAVYAUG-[]_DO-0.1_FL-False_FA-[]_FG-0.0_CB-False_CBB-0.0/model-epoch-6.pt',
                                     )


    print(m)
    print(a)
    print(sum(a) / 7)
    print(f1)
    print(r)



