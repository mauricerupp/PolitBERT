# coding=utf-8
# Copyright 2020 The SqueezeBert authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for SqueezeBERT."""

from ...utils import logging
from ..bert.tokenization_bert import BertTokenizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "squeezebert/squeezebert-uncased": "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/vocab.txt",
        "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/vocab.txt",
        "squeezebert/squeezebert-mnli-headless": "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "squeezebert/squeezebert-uncased": 512,
    "squeezebert/squeezebert-mnli": 512,
    "squeezebert/squeezebert-mnli-headless": 512,
}


PRETRAINED_INIT_CONFIGURATION = {
    "squeezebert/squeezebert-uncased": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli": {"do_lower_case": True},
    "squeezebert/squeezebert-mnli-headless": {"do_lower_case": True},
}


class SqueezeBertTokenizer(BertTokenizer):
    r"""
    Constructs a SqueezeBert tokenizer.

    :class:`~transformers_local.SqueezeBertTokenizer is identical to :class:`~transformers_local.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers_local.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
