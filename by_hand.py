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
    label, text = row
    tokens = [token.text for token in nlp(text)]
    return label, tokens

def load_data(filename: str) -> List[ProcessedRow]:
    """
    TODO: implement
    Read in the file, and use `process` on each line.
    Make sure to use csv.reader!
    """
    with open(filename) as f:
        reader = csv.reader(f)
        return [process(row) for row in tqdm.tqdm(reader)]

# TODO: Load training data, validation data, and test data
training_data = load_data('data/bbc-train.csv')
validation_data = load_data('data/bbc-validate.csv')
test_data = load_data('data/bbc-test.csv')

# TODO: construct mappings idx_to_word and word_to_idx
# Hint: use a set to get unique words, then use `enumerate`
# to get a mapping word <-> index
idx_to_word = list({token
                    for label, tokens in training_data + validation_data + test_data
                    for token in tokens})
word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

# TODO: construct mapping idx_to_label and label_to_idx
idx_to_label = list({label for label, text in training_data + validation_data + test_data})
label_to_idx = {label: idx for idx, label in enumerate(idx_to_label)}

class Model(torch.nn.Module):
    def __init__(self,
                 word_to_idx: Dict[str, int],
                 label_to_idx: Dict[str, int],
                 embedding_dim: int = 100) -> None:
        super().__init__()

        # TODO: store passed in parameters
        # TODO: create a torch.nn.Embedding layer.
        #       need to specify num_embeddings and embedding_dim
        self.word_to_idx = word_to_idx
        self.num_words = len(word_to_idx)
        self.embedding = torch.nn.Embedding(num_embeddings=self.num_words, embedding_dim=embedding_dim)

        # TODO: create a torch.nn.Linear layer
        #       need to specify in_features and out_features
        self.label_to_idx = label_to_idx
        num_labels = len(self.label_to_idx)
        self.linear = torch.nn.Linear(in_features=embedding_dim, out_features=num_labels)

        # TODO: create a loss function that's just torch.nn.CrossEntropyLoss
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                tokens: List[str],
                label: Optional[str] = None) -> Dict[str, torch.Tensor]:

        # TODO: convert the tokens to a tensor of word indices
        inputs = torch.tensor([self.word_to_idx[word] for word in tokens])

        # TODO: use the embedding layer to embed the tokens
        embeddings = self.embedding(inputs)

        # TODO: take the mean of the embeddings along dimension 0 (sequence length)
        encoding = torch.mean(embeddings, dim=0)

        # TODO: pass the encoding through the linear layer to get logits
        logits = self.linear(encoding)
        output = {"logits": logits}

        if label is not None:
            # TODO: find the corresponding label_id and stick it in a 1-D tensor
            # TODO: use .unsqueeze(0) to add a batch dimension to the logits
            # TODO: compute the loss
            label_id = self.label_to_idx[label]
            output["loss"] = self.loss(logits.unsqueeze(0), torch.tensor([label_id]))

        # TODO: return a dict with the logits and (if we have it) the loss
        return output

NUM_EPOCHS = 100

# instantiate the model
model = Model(word_to_idx, label_to_idx, 100)

# instantiate an optimizer
optimizer = torch.optim.Adagrad(model.parameters())


for epoch in range(NUM_EPOCHS):
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
            optimizer.zero_grad()
            # TODO: call the model on the inputs
            output = model(text, label)
            # TODO: pull the loss out of the output
            loss = output["loss"]
            # TODO: add loss.item() to epoch_loss
            epoch_loss += loss.item()
            # TODO: call .backward on the loss
            loss.backward()
            # TODO: step the optimizer
            optimizer.step()

            # TODO: get the (actual) label_id and the predicted label_id
            # hint: use torch.argmanx for the second
            label_id = label_to_idx[label]
            predicted = torch.argmax(output["logits"]).item()

            # TODO: update num_seen and num_correct
            num_seen += 1
            num_correct += (predicted == label_id)
            # TODO: compute accuracy
            accuracy = num_correct / num_seen
            # TODO: add accuracy to the tqdm description
            it.set_description(f"train acc: {accuracy:.3f} loss: {epoch_loss / num_seen:.3f}")

    # Compute validation accuracy

    # TODO: set the model to .eval() mode
    model.eval()

    # set num_correct and num_seen to 0
    num_correct = 0
    num_seen = 0

    with tqdm.tqdm(validation_data) as it:
        validation_loss = 0.0
        for label, text in it:
            # TODO: call the model on the inputs
            output = model(text, label)

            # TODO: compute the actual label_id and the predicted label_id
            label_id = label_to_idx[label]
            predicted = torch.argmax(output["logits"]).item()

            validation_loss += output["loss"].item()

            # TODO: increment counters
            num_seen += 1
            num_correct += (predicted == label_id)
            accuracy = num_correct / num_seen
            it.set_description(f"valid acc: {accuracy:.3f} loss: {validation_loss / num_seen:.3f}")

    # print epoch, epoch loss, accuracy
    print(epoch, epoch_loss, num_correct / num_seen)

# TODO: evaluate accuracy on test dataset
model.eval()
num_correct = 0
num_seen = 0
for label, text in test_data:
    output = model(text, label)
    label_id = label_to_idx[label]
    predicted = torch.argmax(output["logits"]).item()
    num_seen += 1
    num_correct += (predicted == label_id)
    accuracy = num_correct / num_seen
print(accuracy)
