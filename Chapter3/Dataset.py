import torch
from tiktoken import Encoding
from torch.utils.data import Dataset

PADDING_TOKEN = '<|padding|>'

class TextDataSet(Dataset):
    def __init__(self, tokens: list[int], tokenizer: Encoding, sequence_size: int, stride: int):
        self._tokenizer = tokenizer
        self._input_ids = []
        self._target_ids = []

        if len(tokens) < sequence_size:
            padding_value = tokenizer.encode(PADDING_TOKEN, allowed_special={PADDING_TOKEN})[0]
            tokens += [padding_value] * (sequence_size - len(tokens))

        token_length = len(tokens)

        for i in range(0, (token_length - sequence_size) + 1, stride):
            input_chunk = tokens[i:i + sequence_size]
            target_chunk = tokens[i + 1: i + sequence_size + 1]

            self._input_ids.append(torch.tensor(input_chunk))
            self._target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self._input_ids)

    def __getitem__(self, idx):
        return self._input_ids[idx], self._target_ids[idx]