import tiktoken
from torch.utils.data import DataLoader

from Chapter3.Dataset import TextDataSet
from Chapter3.learnable_positional_embedding import LearnablePositionalEncoding
from Chapter3.positional_encoding import PositionalEncoding

input_data = """Still, Prime Minister Benjamin Netanyahu has repeatedly stated that a ground operation in Rafah 
is necessary to keep the pressure on Hamas to release the remaining hostages and to achieve victory. 
As Israel's leadership came closer to a final decision, the US began to review proposed transfers of particular weapons to 
Israel that might be used in Rafah, the US official said. 
The process of carrying out the review began in April and led to the pause in shipments of the two types of bombs.\
"""

tokenizer = tiktoken.get_encoding("cl100k_base")
tokens = tokenizer.encode(input_data)

stride = 1
sequence_size = 12
batch_size = 8
embedding_size = 16

text_data_set = TextDataSet(tokens, tokenizer, sequence_size, stride)

training_data_loader = DataLoader(text_data_set, batch_size=batch_size, shuffle=True, drop_last=True)
x, y = next(iter(training_data_loader))

print(x.shape)
print(y.shape)

# positional encoding with repetitive id
positional_encoding = PositionalEncoding(vocab_size=tokenizer.n_vocab, batch_size=batch_size,
                                         sequence_size=sequence_size,
                                         embedding_dim=embedding_size)

pe_embedding = positional_encoding(x)

# learnable positional encoding
learnable_positional_encoding = LearnablePositionalEncoding(vocab_size=tokenizer.n_vocab,
                                         sequence_size=sequence_size,
                                         embedding_dim=embedding_size)

learnable_pe_embedding = learnable_positional_encoding(x)


