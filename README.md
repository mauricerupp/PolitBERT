# PolitBERT

This is the code of my Master's thesis 'PolitBERT: Deepfake Detection of American Politicians
using Natural Language Processing'.

## Pre-Trained Specialized Language Model
First, the original version of BERT was initialized with the official weights provided by the `huggingface.co` library and then further pre-trained on a novel dataset of English-speaking politicians. This dataset is the biggest publicly available dataset of English-speaking
politicians so far, consisting of 1.5 M sentences from over 1000 persons.
The dataset is available [here](https://www.kaggle.com/mauricerupp/englishspeaking-politicians) and the pre-trained BERT model is available [here](https://huggingface.co/maurice/PolitBERT).

## Fine-Tuning on Author Classification
This pre-trained model was then fine-tuned on the classification of seven impactful American politicians using multiple techniques to handle sever data imbalance, such as data augmentation in the latent space and conventional text augmentation, balanced loss functions, dropout, ensemble learning and undersampling.

The best-performing model achieved a macro-averaged test set accuracy of **74.63%**.
