from typing import Dict, Optional

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

import torch

@Model.register("imdb")
class ImdbModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 label_namespace: str = "labels") -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        hidden_dim = self.encoder.get_output_dim()
        num_classes = vocab.get_vocab_size(label_namespace)

        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)

        # Loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # Track accuracy
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                review: Dict[str, torch.Tensor],
                sentiment: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded_review = self.text_field_embedder(review)

        mask = get_text_field_mask(review)
        encoded_review = self.encoder(embedded_review, mask)
        logits = self.linear(encoded_review)

        output = {"logits": logits}

        if sentiment is not None:
            output["loss"] = self.loss(logits, sentiment)
            self.accuracy(logits, sentiment)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
