from typing import Dict, Optional

from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

import torch

@Model.register("bbc")
class BBCModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 dropout: float = 0.0) -> None:
        super().__init__(vocab)
        self.text_field_embedder = text_field_embedder
        self.encoder = encoder

        hidden_dim = self.encoder.get_output_dim()
        num_classes = vocab.get_vocab_size("labels")

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(in_features=hidden_dim, out_features=num_classes)

        # Loss function
        self.loss = torch.nn.CrossEntropyLoss()

        # Track accuracy
        self.metrics = {
            "acc1": CategoricalAccuracy(),
            "acc3": CategoricalAccuracy(top_k=3)
        }

    def forward(self,
                text: Dict[str, torch.Tensor],
                category: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.text_field_embedder(text)

        mask = get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)
        logits = self.linear(self.dropout(encoded_text))

        num_categories = self.vocab.get_vocab_size("labels")
        category_names = [[self.vocab.get_token_from_index(i, namespace="labels") for i in range(num_categories)]]

        output = {"logits": logits, "category_names": category_names}

        if category is not None:
            output["loss"] = self.loss(logits, category)
            for metric in self.metrics.values():
                metric(logits, category)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: metric.get_metric(reset)
                for name, metric in self.metrics.items()}
