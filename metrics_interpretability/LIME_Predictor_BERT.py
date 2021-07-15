import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from scipy.special import softmax
import numpy as np
from lime.lime_text import LimeTextExplainer
import re


class LIME_Prediction:

    def __init__(self, model_path, seq_length):

        self.model, self.tokenizer, self.model_config = self.load_model(model_path)
        self.max_seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_model(self, model_path):

        model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              num_labels=7,
                                                              output_attentions=False,
                                                              output_hidden_states=False)
        model.load_state_dict(torch.load(model_path))
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig

        return model, tokenizer, config

    def predict_label(self, text_a):

        self.model.to(self.device)

        input_ids, input_mask, segment_ids = self.convert_text_to_features(text_a)
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids, input_mask)

        logits = outputs[0].detach().cpu().numpy()
        logits = softmax(logits, axis=1)
        logits_label = np.argmax(logits).flatten()
        label = logits_label

        logits_confidence = logits[0][logits_label]

        return label, logits_confidence

    def _truncate_seq_pair(self, tokens_a, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        while True:
            total_length = len(tokens_a)
            if total_length <= max_length:
                break
            if len(tokens_a) > max_length:
                tokens_a.pop()

    def convert_text_to_features(self, text_a):

        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        sequence_a_segment_id = 0
        cls_token_segment_id = 1
        pad_token_segment_id = 0
        mask_padding_with_zero = True
        pad_token = 0
        tokens_a = self.tokenizer.tokenize(text_a)

        self._truncate_seq_pair(tokens_a, self.max_seq_length - 2)

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        #
        # # Zero-pad up to the sequence length.
        padding_length = self.max_seq_length - len(input_ids)

        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length

        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)

        return input_ids, input_mask, segment_ids

    def predictor(self, text):

        examples = []
        for example in text:
            examples.append(self.convert_text_to_features(example))

        results = []
        for example in examples:

            with torch.no_grad():
                outputs = self.model(example[0], example[1], example[2])

            logits = outputs[0].detach().cpu().numpy()
            logits = softmax(logits, axis=1)
            results.append(logits[0])

        results_array = np.array(results)

        return results_array


if __name__ == '__main__':

    model_path = '/data/cvg/maurice/logs/0_Network_Backups/Final_Models/Multiclass/PolitBERT_Finetuning_cleaned_inclVAL_LR-2e-05_BS-16_L2-0_EDA-True/model-epoch-6.pt'
    prediction = LIME_Prediction(model_path, seq_length = 512)
    label_names = ['Clinton', 'Obama', 'Pence', 'Biden', 'Sanders', 'Trump', 'Harris']
    explainer = LimeTextExplainer(class_names=label_names)

    train_ls = ['do you believe that i would vote for barack obama?']

    for example in train_ls:

        exp = explainer.explain_instance(example, prediction.predictor, labels=[4, 5])
        exp.save_to_file('/data/cvg/maurice/logs/lime/{}.html'.format(re.sub(' ', '', example[:-2])), labels=(4,5))