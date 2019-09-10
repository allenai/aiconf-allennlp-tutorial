from typing import Optional, Dict, List, Tuple
import csv
import random

import spacy
import torch
import tqdm

# Load the spacy model
nlp = spacy.load('en_core_web_sm')

# Just some type aliases to make things cleaner
RawRow = List[str]
ProcessedRow = Tuple[str, List[str]]

def process(row: RawRow) -> ProcessedRow:
    """
    TODO: implement
    row is [category, text]
    want to return (category, list_of_tokens)
    use spacy ("nlp") to tokenize text, and then
    use token.text to get just the string tokens.
    """
    pass

def load_data(filename: str) -> List[ProcessedRow]:
    """
    TODO: implement
    Read in the file, and use `process` on each line.
    Make sure to use csv.reader!
    """
    pass

# TODO: Load training data, validation data, and test data
training_data = ...
validation_data = ...
test_data = ...

# TODO: construct mappings idx_to_word and word_to_idx
# Hint: use a set to get unique words, then use `enumerate`
# to get a mapping word <-> index
idx_to_word = ...
word_to_idx = ...

# TODO: construct mapping idx_to_label and label_to_idx
idx_to_label = ...
label_to_idx = ...

class Model(torch.nn.Module):
    def __init__(self,
                 word_to_idx: Dict[str, int],
                 label_to_idx: Dict[str, int],
                 embedding_dim: int = 100) -> None:
        super().__init__()

        # TODO: store passed in parameters

        # TODO: create a torch.nn.Embedding layer.
        #       need to specify num_embeddings and embedding_dim

        # TODO: create a torch.nn.Linear layer
        #       need to specify in_features and out_features

        # TODO: create a loss function that's just torch.nn.CrossEntropyLoss

    def forward(self,
                tokens: List[str],
                label: Optional[str] = None) -> Dict[str, torch.Tensor]:

        # TODO: convert the tokens to a tensor of word indices

        # TODO: use the embedding layer to embed the tokens

        # TODO: take the mean of the embeddings along dimension 0 (sequence length)

        # TODO: pass the encoding through the linear layer to get logits

        if label is not None:
            # TODO: find the corresponding label_id and stick it in a 1-D tensor
            # TODO: use .unsqueeze(0) to add a batch dimension to the logits
            # TODO: compute the loss
            pass

        # TODO: return a dict with the logits and (if we have it) the loss
        pass

NUM_EPOCHS = 100

# instantiate the model
model = Model(word_to_idx, label_to_idx, 100)

# instantiate an optimizer
optimizer = torch.optim.Adagrad(model.parameters())


for epoch in range(NUM_EPOCHS):
    print(f"epoch {epoch}")
    # shuffle the training data
    random.shuffle(training_data)

    epoch_loss = 0.0
    num_correct = 0
    num_seen = 0

    with tqdm.tqdm(training_data) as it:
        # Set the model in train mode
        model.train()

        for label, text in it:
            # TODO: zero out the gradients
            # TODO: call the model on the inputs
            # TODO: pull the loss out of the output
            # TODO: add loss.item() to epoch_loss
            # TODO: call .backward on the loss
            # TODO: step the optimizer

            # TODO: get the (actual) label_id and the predicted label_id
            # hint: use torch.argmax for the second

            # TODO: update num_seen and num_correct
            # TODO: compute accuracy
            # TODO: add accuracy and loss to the tqdm description
            it.set_description(f"")

    # Compute validation accuracy

    # TODO: set the model to .eval() mode

    # set num_correct and num_seen to 0
    num_correct = 0
    num_seen = 0

    with tqdm.tqdm(validation_data) as it:
        for label, text in it:
            # TODO: call the model on the inputs

            # TODO: compute the actual label_id and the predicted label_id

            # TODO: increment counters

            # TODO: add accuracy and loss to the tqdm description
            it.set_description(f"")


# TODO: evaluate accuracy on test dataset
