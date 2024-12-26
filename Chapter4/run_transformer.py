import tiktoken
import torch
from tiktoken import Encoding

from Chapter3.Dataset import PADDING_TOKEN
from Chapter4.transformer_component import TransformerConfig, Transformer

input_data = "I don’t know, but I do know who will have an answer for us."

def encode(tokenizer: Encoding, candidate_text: str, window_size: int, padding_value: int):
    tokens = tokenizer.encode(candidate_text)

    if len(tokens) >= window_size:
        return torch.tensor(tokens[-window_size:]).unsqueeze(0)

    if len(tokens) < window_size:
        tokens += [padding_value] * (window_size - len(tokens))

    return torch.tensor(tokens).unsqueeze(0)

def run_transformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_size=32
    cl100k_base = tiktoken.get_encoding("cl100k_base")
    tokenizer_size = cl100k_base.n_vocab
    padding_value = tokenizer_size+1

    tokenizer = tiktoken.Encoding(
        name="cl100k_im",
        pat_str=cl100k_base._pat_str,
        mergeable_ranks=cl100k_base._mergeable_ranks,
        special_tokens={
            **cl100k_base._special_tokens,
            PADDING_TOKEN: tokenizer_size+1,
        },
    )

    config = TransformerConfig(embed_dim=768, vocab_size=tokenizer.n_vocab,
                               attention_head_size=4, attention_layer_size=12,
                               hidden_dropout_prob=0.1, window_size=window_size,
                               inference_mode=True, device=device)

    transformer = Transformer(config,)
    transformer.to(device)

    sample_text = input_data
    spit_count = 5
    for i in range(spit_count):
        tokens = encode(tokenizer, sample_text, window_size, padding_value).to(device)
        logits = transformer(tokens)
        print('logits shape', logits.shape)
        logits = logits[:, -1, :]

        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        result = tokenizer.decode(idx_next.squeeze(0).tolist())

        sample_text = sample_text+result
        print(sample_text)

if __name__ == '__main__':
    run_transformer()