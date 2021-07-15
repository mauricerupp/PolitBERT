import torch
import csv
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import os
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import matplotlib.pyplot as plt
import logging
import re
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import random
from nltk.tokenize import RegexpTokenizer
from scipy.special import softmax

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# ---------- Samplers ------------ #
def get_weights_for_each_sample(dataset, bert_dataset=False):
    # get the distribution of classes in the dataset
    label_to_count = {}
    for idx in range(dataset.__len__()):
        if bert_dataset:
            label = dataset.__getitem__(idx)['label'].item()
        else:
            label = dataset.__getitem__(idx)[1]
        if label in label_to_count:
            label_to_count[label] += 1
        else:
            label_to_count[label] = 1

    # weight for each sample
    if bert_dataset:
        weights = [1.0 / label_to_count[dataset.__getitem__(idx)['label'].item()] for idx in range(dataset.__len__())]
    else:
        weights = [1.0 / label_to_count[dataset.__getitem__(idx)[1]] for idx in range(dataset.__len__())]

    return torch.DoubleTensor(weights)


def get_sampler(weights, y, oversampling):
    if oversampling:
        train_sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights),
                                              replacement=True)
    # if we don't oversample, we take the samples with the label which has less occurences only once
    # and train it against a random subset of samples with the other labels
    else:
        # get the amount of samples of the least common class
        samples_count = Counter(y)
        amount_of_least_samples = samples_count.most_common()[-1][1]
        # sample the amount of the least samples times the amount of classes per epoch
        train_sampler = WeightedRandomSampler(weights=weights,
                                              num_samples=amount_of_least_samples * len(samples_count),
                                              replacement=False)
    return train_sampler


# ---------- Model Training ------------ #
def linear_schedule(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def process_predict_batch(model, multiclass, sample):
    texts = sample[0].to(device)
    labels = sample[1].to(device)
    if not multiclass:
        labels = labels.float()
    # Run predictions if packing is done or not
    if len(sample) == 3:
        sentence_lengths = sample[2].to(device)
        output = model(texts, sentence_lengths)
    else:
        output = model(texts)
    if not multiclass:
        output = torch.squeeze(output)
    return labels, output


def train_epoch(model, train_dataloader, optimizer, loss_fn, multiclass):
    losses = []
    n_correct = 0
    all_labels = []
    preds = []
    # Iterate mini batches over training dataset
    for sample in train_dataloader:
        # process and predict the sample
        labels, output = process_predict_batch(model, multiclass, sample)
        all_labels.extend(labels)
        # Set gradients to zero
        optimizer.zero_grad()
        # Compute loss
        loss = loss_fn(output, labels)
        # Backpropagate (compute gradients)
        loss.backward()
        # Make an optimization step (update parameters)
        optimizer.step()
        # Log metrics
        losses.append(loss.item())
        if multiclass:
            n_correct += torch.sum(output.argmax(1) == labels).item()
            preds.extend(output.argmax(1))
        else:
            n_correct += torch.sum(torch.round(output) == labels).item()
            preds.extend(torch.round(output))

    # get the per-class-accuracy
    acc_per_class, accuracy = create_accuracies(all_labels, multiclass, preds)
    return np.mean(np.array(losses)), accuracy, acc_per_class


def evaluate(model, dataloader, loss_fn, multiclass):
    losses = []
    n_correct = 0
    all_labels = []
    preds = []
    with torch.no_grad():
        for sample in dataloader:
            labels, output = process_predict_batch(model, multiclass, sample)

            all_labels.extend(labels)
            # Compute loss
            loss = loss_fn(output, labels)
            # Save metrics
            losses.append(loss.item())
            if multiclass:
                n_correct += torch.sum(output.argmax(1) == labels).item()
                preds.extend(output.argmax(1))
            else:
                n_correct += torch.sum(torch.round(output) == labels).item()
                preds.extend(torch.round(output))

    # get the per-class-accuracy
    acc_per_class, accuracy = create_accuracies(all_labels, multiclass, preds)

    return np.mean(np.array(losses)), accuracy, acc_per_class


def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_fn, general_logpath, model_logpath, modelname, multiclass=False):

    # setup the tensorboard and create a folder for the tb files
    TB_LOG_PATH = general_logpath + '/tb_logs_nonBERT'
    if not os.path.exists(TB_LOG_PATH):
        os.makedirs(TB_LOG_PATH)
    tb_writer = SummaryWriter(log_dir=TB_LOG_PATH)

    # We will monitor the loss functions as the training progresses
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0

    for epoch in range(n_epochs):
        model.train()

        train_loss, train_accuracy, train_acc_per_class = train_epoch(model, train_dataloader, optimizer, loss_fn, multiclass)
        model.eval()
        val_loss, val_accuracy, val_acc_per_class = evaluate(model, val_dataloader, loss_fn, multiclass)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        # add the metrics to the tensorboard
        tb_writer.add_scalars("Loss/train", {str(modelname): train_loss}, epoch)
        tb_writer.add_scalars("Loss/val", {modelname: val_loss}, epoch)
        tb_writer.add_scalars("Acc/train", {modelname: train_accuracy}, epoch)
        tb_writer.add_scalars("Acc/val", {modelname: val_accuracy}, epoch)

        # save the model if the validation accuracy increased
        if epoch != 0:
            if val_accuracies[-1] > best_val_acc:
                torch.save(model, model_logpath + 'weights.pt')
                best_val_acc = val_accuracies[-1]
        epoch_summary = 'Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}%, train LM_acc per class: {}, \nval_loss: {:.4f}, val_accuracy: {:.4f}%, val LM_acc per class: {}'.format(epoch+1, n_epochs,
                                                                                                      train_losses[-1],
                                                                                                      train_accuracies[-1],
                                                                                                      train_acc_per_class,
                                                                                                      val_losses[-1],
                                                                                                      val_accuracies[-1],
                                                                                                      val_acc_per_class)
        print(epoch_summary)
        logging.basicConfig(filename=model_logpath + 'modellog.log', level=logging.INFO)
        logging.info(epoch_summary)

    tb_writer.flush()
    return train_losses, val_losses, train_accuracies, val_accuracies


# --------- Language Model Training --------------#
def LM_acc(preds, labels):
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    return np.sum(preds == labels) / len(labels)


def process_predict_LM_batch(model, batch, all_predictions, all_labels, return_probabilities=False):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    all_predictions.extend(np.argmax(outputs[1].detach().cpu().numpy(), axis=1).flatten())
    all_labels.extend(labels.detach().cpu().numpy().flatten())

    if return_probabilities:
        return outputs, labels, all_predictions, all_labels, softmax(outputs[1].detach().cpu().numpy(), axis=1)

    else:
        return outputs, labels, all_predictions, all_labels


def train_LM_epoch(model, train_loader, optimizer, loss_fn, lr_scheduler=None):
    model.train()
    total_train_acc = 0
    total_train_loss = 0
    all_labels = []
    all_preds = []

    for batch in train_loader:
        outputs, labels, all_preds, all_labels = process_predict_LM_batch(model, batch, all_preds, all_labels)

        optimizer.zero_grad()

        if loss_fn is not None:
            loss = loss_fn(outputs[1].to(device, dtype=torch.float32), labels)
        else:
            loss = outputs[0]

        loss.sum().backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        total_train_loss += loss.sum().item()
        total_train_acc += LM_acc(outputs[1], labels)

    matrix = confusion_matrix(all_labels, all_preds)
    acc_per_class = matrix.diagonal() / matrix.sum(1)
    return acc_per_class, total_train_loss


def evaluate_LM(model, val_loader, loss_fn):
    model.eval()
    total_val_acc = 0
    total_val_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in val_loader:
            outputs, labels, all_preds, all_labels = process_predict_LM_batch(model, batch, all_preds, all_labels)

            if loss_fn is not None:
                loss = loss_fn(outputs[1].to(device, dtype=torch.float32), labels)
            else:
                loss = outputs[0]

            total_val_loss += loss.sum().item()
            total_val_acc += LM_acc(outputs[1], labels)

        matrix = confusion_matrix(all_labels, all_preds)
        acc_per_class = matrix.diagonal() / matrix.sum(1)

        return acc_per_class, total_val_loss


def train_LM(epochs, model, train_loader, test_loader, optimizer, general_logpath, model_logpath, modelname, loss_fn, single_author=None, val_loader=None):
    # setup the tensorboard
    TB_LOG_PATH = general_logpath + '/tb_logs'
    if not os.path.exists(TB_LOG_PATH):
        os.makedirs(TB_LOG_PATH)
    tb_writer = SummaryWriter(log_dir=TB_LOG_PATH + '/' + modelname)

    best_test_acc = 0
    for epoch in range(epochs):

        print('Training epoch {}'.format(epoch + 1))
        train_acc_per_class, total_train_loss = train_LM_epoch(model, train_loader, optimizer, loss_fn)
        print('Epoch Acc: {}, Epoch Loss: {}, Acc per class: {}'.format(
            np.sum(train_acc_per_class) / len(train_acc_per_class), total_train_loss, train_acc_per_class))

        # val_acc_per_class, val_loss = evaluate_LM(model, val_loader)
        # print('Validation Acc: {}, Acc per class: {}, Loss: {}'.format(np.sum(val_acc_per_class) / len(val_acc_per_class),
        #                                               val_acc_per_class, val_loss))

        # run the model for our test class
        test_acc_per_class, test_loss = evaluate_LM(model, test_loader, loss_fn)
        print('Test Acc: {}, Acc per class: {}, Loss: {}'.format(np.sum(test_acc_per_class) / len(test_acc_per_class),
                                                                 test_acc_per_class, test_loss))


        # write the stuff to the tensorboard
        if type(single_author) == int:
            tb_writer.add_scalar("Single_Loss/train", total_train_loss, epoch)
            tb_writer.add_scalar("Single_Loss/test", test_loss, epoch)
            tb_writer.add_scalar("Single_Acc/train", np.sum(train_acc_per_class) / len(train_acc_per_class), epoch)
            tb_writer.add_scalar("Single_Acc/test", np.sum(test_acc_per_class) / len(test_acc_per_class), epoch)
            tb_writer.add_scalar("Authoracc_{}/test".format(single_author), test_acc_per_class[1], epoch)
            tb_writer.add_scalar("Authoracc_{}/train".format(single_author), train_acc_per_class[1], epoch)

        else:
            tb_writer.add_scalar("Loss/train", total_train_loss, epoch)
            tb_writer.add_scalar("Loss/test", test_loss, epoch)
            tb_writer.add_scalar("Acc/train", np.sum(train_acc_per_class) / len(train_acc_per_class), epoch)
            tb_writer.add_scalar("Acc/test", np.sum(test_acc_per_class) / len(test_acc_per_class), epoch)
            tb_writer.add_scalar("Authoracc_0/test", test_acc_per_class[0], epoch)
            tb_writer.add_scalar("Authoracc_0/train", train_acc_per_class[0], epoch)
            tb_writer.add_scalar("Authoracc_1/test", test_acc_per_class[1], epoch)
            tb_writer.add_scalar("Authoracc_1/train", train_acc_per_class[1], epoch)
            tb_writer.add_scalar("Authoracc_2/test", test_acc_per_class[2], epoch)
            tb_writer.add_scalar("Authoracc_2/train", train_acc_per_class[2], epoch)
            tb_writer.add_scalar("Authoracc_3/test", test_acc_per_class[3], epoch)
            tb_writer.add_scalar("Authoracc_3/train", train_acc_per_class[3], epoch)
            tb_writer.add_scalar("Authoracc_4/test", test_acc_per_class[4], epoch)
            tb_writer.add_scalar("Authoracc_4/train", train_acc_per_class[4], epoch)
            tb_writer.add_scalar("Authoracc_5/test", test_acc_per_class[5], epoch)
            tb_writer.add_scalar("Authoracc_5/train", train_acc_per_class[5], epoch)
            tb_writer.add_scalar("Authoracc_6/test", test_acc_per_class[6], epoch)
            tb_writer.add_scalar("Authoracc_6/train", train_acc_per_class[6], epoch)

        tb_writer.flush()

        if not single_author:
            with open(general_logpath + 'metrics_BERTS.txt', 'a') as f:
                f.write('{}\n'.format(modelname))
                f.write('Overall Test Acc at epoch {}: {}, Acc per class: {}'.format(epoch + 1,
                                                                                     np.sum(test_acc_per_class) / len(
                                                                                         test_acc_per_class),
                                                                                     test_acc_per_class))
                f.close()
        else:
            with open(general_logpath + 'metrics_normalBERT_Single.txt', 'a') as f:
                f.write('{}\n'.format(modelname))
                f.write('Overall Test Acc at epoch {}: {}, Acc per class: {}'.format(epoch + 1,
                                                                                     np.sum(test_acc_per_class) / len(
                                                                                         test_acc_per_class),
                                                                                     test_acc_per_class))
                f.close()

        if np.sum(test_acc_per_class) / len(test_acc_per_class) > best_test_acc:
            best_test_acc = np.sum(test_acc_per_class) / len(test_acc_per_class)
            torch.save(model.module.state_dict(), model_logpath + 'model-epoch-{}.pt'.format(epoch))


# --------- Dataset Helpers --------------#
def remove_contractions_from_sentence(string):
    with open('english_contractions.csv', newline='') as f:
        contraction_list = [tuple(line) for line in csv.reader(f)]
        for contraction, meaning in contraction_list:
            string = string.replace(contraction, meaning)
        return string


def remove_stopwords_from_sentence(string):
    stop_words = ['the', 'to', 'and', 'a', 'is', 'that', 'in']
    #stop_words = set(stopwords.words('english'))
    stop_words = ''.join([word + '|' for word in stop_words])
    stop_word_pattern = '\\b(' + stop_words[:-1] + ')\\b'
    stop_word_pattern = re.compile(stop_word_pattern)
    return re.sub(stop_word_pattern, '', string)


def word_counter(sentence):
    punct_remover = RegexpTokenizer(r'\w+')
    """
    helper function to be able to tokenize the words of each sample in the dataframe
    :param sentence:
    :return:
    """
    return len(punct_remover.tokenize(sentence))


def pad_sentence(sentence, sentence_size, pre_padding):
    n_padding = sentence_size - len(sentence)
    padding = sentence.new(n_padding).fill_(0)
    if pre_padding:
        sentence = torch.cat((padding, sentence), dim=0)
    else:
        sentence = torch.cat((sentence, padding), dim=0)
    return sentence


def unk_initializer(tensor):
    """
    returns a tensor of the same shape as the input in the range [-1, 1]
    :param tensor:
    :return:
    """
    tensor = tensor.uniform_(to=2) - 1
    return tensor


# --------- Data Augmentation --------------#
pattern_one_comma = re.compile('^([^,]+,[^,]+){1}$')
pattern_two_commas = re.compile('^([^,]+,[^,]+){2}$')


def space_counter(string):
    return string.count(' ')


def string_augmentator(modification_prob_one_comma, modification_prob_two_commas, string):
    # case 1: string contains exactly one comma - we take the longer part of the sentences
    if re.search(pattern_one_comma, string):
        if random.uniform(0, 1) < modification_prob_one_comma:
            # split the string where the comma is
            temp = string.split(',')
            longer_part = max(temp, key=word_counter)
            # do not modify the string, if the longer part would be too short to contain useful info
            if not word_counter(longer_part) > 3:
                return string
            # catch the case, where the first part of the sentence is taken and there is no punctuation in the end
            if not re.search(r'[:!?.]$', longer_part):
                longer_part += '.'
            # remove leading spaces
            longer_part = re.sub(r'^\s', '', longer_part)
            # Make the first letter capital
            return longer_part[0].upper() + longer_part[1:]

    # case 2: string contains exactly two commas, remove the part between the commas
    elif re.search(pattern_two_commas, string):
        if random.uniform(0, 1) < modification_prob_two_commas:
            separated_string = string.split(',')
            # if one part is too small we return the whole string since chances of bad sentences are too high
            if not word_counter(separated_string[0]) > 2 or not word_counter(separated_string[2]) > 2:
                return string

            return separated_string[0] + separated_string[2]

    return string


# --------- Metrics --------------#
def create_accuracies(all_labels, multiclass, preds):
    preds = torch.stack(preds).cpu().detach().numpy()
    all_labels = torch.stack(all_labels).cpu().detach().numpy()
    if not multiclass:
        preds = np.squeeze(preds)
        preds = preds.astype(int)
    if multiclass:
        matrix = confusion_matrix(all_labels, preds, labels=[0, 1, 2, 3, 4, 5, 6])
    else:
        matrix = confusion_matrix(all_labels, preds)
    acc_per_class = matrix.diagonal() / matrix.sum(1)
    accuracy = 100.0 * (np.sum(acc_per_class) / len(acc_per_class))
    return acc_per_class, accuracy


def plot_curves(training_curves, logpath):
    # create the folder for all the plots
    curve_folder = logpath + 'curves/'
    if not os.path.exists(curve_folder):
        os.makedirs(curve_folder)

    keys = []
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[1])), v[1])
        keys.append(k)
    plt.title('Validation loss for different models')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(keys)
    plt.savefig(curve_folder + 'val_loss.jpeg')
    plt.close()

    plt.figure()
    keys = []
    for k, v in sorted(training_curves.items()):
        plt.plot(np.arange(len(v[3])), v[3])
        keys.append(k)
    plt.title('Validation accuracy for different models')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(keys)
    plt.savefig(curve_folder + 'val_acc.jpeg')
    plt.close()


def create_metrics(model, dataloader, multiclass, model_logpath):
    labels = []
    output = []
    with torch.no_grad():
        for samples in dataloader:
            texts_batch = samples[0].to(device)
            labels_batch = samples[1].to(device)
            labels.extend(labels_batch)
            # Run predictions
            if len(samples) == 3:
                text_lengths = samples[2].to(device)
                out = model(texts_batch, text_lengths)
            else:
                out = model(texts_batch)

            if not multiclass:
                output.extend(torch.round(out))
            else:
                output.extend(out.argmax(1))

        output = torch.stack(output).cpu().detach().numpy()
        labels = torch.stack(labels).cpu().detach().numpy()
        if not multiclass:
            output = np.squeeze(output)
            output = output.astype(int)

    # create the confusion matrix
    if multiclass:
        matrix = confusion_matrix(labels, output, labels=[0,1,2,3,4,5,6])
        prec, rec, f1_score, support = precision_recall_fscore_support(labels, output, labels=[0,1,2,3,4,5,6])
    else:
        matrix = confusion_matrix(labels, output)
        prec, rec, f1_score, support = precision_recall_fscore_support(labels, output)

    acc_per_class = matrix.diagonal()/matrix.sum(1)
    metrics = "Confusion Matrix:\n{}\n Precision:\n{}\n Recall:\n{}\n F1:\n{}\n Support:\n{}\n Acc. per Class:\n{}\n".format(matrix, prec, rec, f1_score, support, acc_per_class)

    print(metrics)
    logging.basicConfig(filename=model_logpath + 'modellog.log', level=logging.INFO)
    logging.info(metrics)
