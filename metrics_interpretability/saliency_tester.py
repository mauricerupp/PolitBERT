from torchnlp.encoders.text import StaticTokenizerEncoder
from nltk.tokenize import RegexpTokenizer
from utilities import *
import data_handler
from prettytable import PrettyTable
import seaborn as sns
from collections import Counter

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def highest_gradients_words_visualizer():
    most_common_max_pos = [('going', 1552), ('people', 1160), ('get', 945), ('know', 743), ('one', 704), ('president', 661), ('think', 638), ('well', 566), ('us', 557), ('make', 513), ('said', 435), ('time', 433), ('look', 423), ('need', 340), ('want', 324), ('got', 323), ('like', 321), ('america', 305), ('go', 305), ('would', 298)]

    most_common_max_neg = [('people', 16074), ('going', 14747), ('know', 10134), ('think', 9369), ('want', 8004), ('us', 7743), ('one', 6206), ('get', 5923), ('well', 5698), ('would', 5221), ('said', 4883), ('time', 4745), ('like', 4738), ('country', 4344), ('lot', 4267), ('right', 4179), ('great', 4128), ('got', 3581), ('go', 3504), ('much', 3431)]

    x1, y1 = zip(*most_common_max_pos)
    x2, y2 = zip(*most_common_max_neg)

    # plot the sentence length chart
    fig, ax = plt.subplots()
    ax.bar(x1, y1, .3, label='Positive Classification Gradients', linewidth='0', align='edge')
    ax.bar(x2, y2, -.3, label='Negative Classification Gradients', linewidth='0', align='edge')
    ax.legend()
    plt.ylabel("# Occurences")
    plt.xlabel(
        "Words under the top 2 highest argmax gradients at classification.\n Negative vs. Positive Classification for Joe Biden")
    plt.show()


def sentence_preprocessor(sentence, lookup_table, punct_remover):
    # remove contractions and to lower case
    sentence = remove_contractions_from_sentence(sentence.lower())
    # tokenize the sentence
    sentence = punct_remover.tokenize(sentence)
    sentence = lookup_table.encode(' '.join(sentence)).long()
    sentence[sentence==1] = 0
    sentence = data_handler.pad_sentence(sentence, pre_padding=False, sentence_size=16)
    return sentence


def plot_saliency_heatmap(words, values):
    ax = sns.heatmap(values, xticklabels=50)
    ax.set_yticklabels(words)
    plt.yticks(rotation=360)

    ax2 = ax.twinx()
    ax2.set_yticks(ax.yaxis.get_ticklocs())
    ax2.set_yticklabels(np.flip(np.around(np.sum(values, axis=1), decimals=3), axis=0))

    ax2.set_ylabel('Gradient Sum')
    ax.set_ylabel('Word')
    ax.set_xlabel('Embedding Position')
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.show()
    plt.close()


def get_saliency_of_sentence(sentence, model, lookup_table, punct_remover):
    sample = sentence_preprocessor(sentence, lookup_table, punct_remover)
    sample = sample[None]
    # Run the prediction and get the saliency
    prediction = model(sample)
    prediction.backward()
    print("Given test sentence:\n\"{}\"".format(sentence))
    print("The model predicted a class value of {:.4f}, which is assigned to class {:.0f}.\n".format(prediction.item(),
                                                                                                     torch.round(
                                                                                                         prediction).item()))

    gradients = model.embedding.weight.grad.data.abs()
    word_indices = sample.squeeze()
    # remove all padded and unknown words
    word_indices = word_indices[word_indices!=0]
    word_gradients = gradients[word_indices]

    plot_saliency_heatmap([lookup_table.decode([word]) for word in word_indices], word_gradients.numpy())

    saliency_summed = torch.sum(word_gradients, dim=1)
    saliency_max, _ = torch.max(word_gradients, dim=1)

    print("Saliency Maps:")
    t = PrettyTable(['Word', 'Summed up Gradient'])
    for i in range(len(word_indices)):
        index = word_indices[i].numpy()
        t.add_row([lookup_table.decode([index]), str(saliency_summed[i].item())[:6]])
    print(t)

    t = PrettyTable(['Word', 'Max Gradient'])
    for i in range(len(word_indices)):
        index = word_indices[i].numpy()
        t.add_row([lookup_table.decode([index]), str(saliency_max[i].item())[:6]])
    print(t)


def create_saliency_overview(model, data, vocabulary):
    """
    Used to determine which words are the most important while doing positive classification
    :param model:
    :param data:
    :param packing:
    """
    model.embedding.requires_grad_(requires_grad=True)
    model.eval()
    most_impactful_pos_words = Counter()
    most_impactful_neg_words = Counter()
    # Iterate sample by sample over training data to have the correct gradients for each sample
    for sample in data:
        text = sample[0].to(device)
        text = text[None]

        # Run prediction for this sentence
        if len(sample) == 3:
            sentence_length = sample[2].to(device)
            output = model(text, sentence_length)
        else:
            output = model(text)
        # if its a positive prediction, we calculate the gradients

        output.backward()
        gradients = model.embedding.weight.grad.data.abs()

        word_indices = text.squeeze()
        # remove all padded and unknown words
        word_indices = word_indices[word_indices != 0]
        word_gradients = gradients[word_indices]
        # get the saliency of each word in the sentence
        saliency_max, _ = torch.max(word_gradients, dim=1)
        # get the two words with the highest summed gradient and add them to a counter
        arg_highest_grad = torch.argmax(saliency_max).item()
        if torch.round(output) == 1:
            try:
                most_impactful_pos_words[vocabulary.decode([(word_indices[arg_highest_grad].item())])] += 1
            except RuntimeError:
                pass
        else:
            try:
                most_impactful_neg_words[vocabulary.decode([(word_indices[arg_highest_grad].item())])] += 1
            except RuntimeError:
                pass

        # remove the highest gradient from the saliencies and the words
        saliency_max = torch.cat([saliency_max[:arg_highest_grad],
                                     saliency_max[arg_highest_grad + 1:]])

        word_indices = torch.cat([word_indices[:arg_highest_grad], word_indices[arg_highest_grad + 1:]])
        # add the second highest summed gradient to the list
        if torch.round(output) == 1:
            try:
                most_impactful_pos_words[vocabulary.decode([(word_indices[torch.argmax(saliency_max).item()].item())])] += 1
            except RuntimeError:
                pass
        else:
            try:
                most_impactful_neg_words[vocabulary.decode([(word_indices[torch.argmax(saliency_max).item()].item())])] += 1
            except RuntimeError:
                pass

    print("Most impactful words with positive classification using max:")
    print(most_impactful_pos_words.most_common(20))
    print("Most impactful words with negative classification using max:")
    print(most_impactful_neg_words.most_common(20))


if __name__ == "__main__":
    punct_remover = RegexpTokenizer(r'\w+')
    # read the vocabulary and create a lookup table
    with open('data_MA/vocabulary.txt', 'r') as filehandle:
        voc = [current_place.rstrip() for current_place in filehandle.readlines()]

    lookup_table = StaticTokenizerEncoder(voc[5:])
    # Prepare the model
    MODEL_WEIGHTS = 'data_MA/weights.pt'
    model = torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu'))
    model.embedding.requires_grad_(requires_grad=True)
    model.eval()

    # print the prediction and saliency of the sentence:
    test_sentence = 'Donald Trump is a bad person'

    get_saliency_of_sentence(test_sentence, model, lookup_table, punct_remover)
