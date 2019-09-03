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
                 ...) -> None:
        super().__init__(vocab)

        # 1. We want the constructor to accept a TextFieldEmbedder
        # and we need to save it as a class variable.

        # 2. We want the constructor to accept a Seq2VecEncoder
        # and we need to save it as a class variable

        # 3. We need to construct the final linear layer, it should have
        #   in_features = the output dimension of the Seq2VecEncoder
        #   out_features = the number of classes we're predicting
        # We can get the latter from the "labels" namespace of the vocabulary

        # 4. We also need to instantiate a loss function for our model.
        # Here we'll just use PyTorch's built in cross-entropy loss
        # https://pytorch.org/docs/stable/nn.html#crossentropyloss

        # 5. Finally, we want to track some metrics as we train, at the very
        # least, categorical accuracy:
        # https://allenai.github.io/allennlp-docs/api/allennlp.training.metrics.html#categorical-accuracy
        # store them in a dictionary so that `self.get_metrics` works correctly.


    def forward(self, ...) -> Dict[str, torch.Tensor]:
        # Our forward function needs to take arguments that correspond
        # to the fields of our instance.

        # 1. In this case we'll always have a "text" input

        # 2. and we'll sometimes have a "category" input
        #    (so it should have a None default value for when we're doing inference)

        # 3. our first step should be to apply our TextFieldEmbedder to the text input

        # 4. then we should apply our Seq2VecEncoder to the embedded text
        #    We'll need to provide a _mask_ which we can get from util.get_text_field_mask

        # 5. we can then apply apply our linear layer to the encoded text to get
        #    the logits corresponding to the predicted class probabilities

        # 6. our outputs need to be in a dict, so create one that contains the logits

        # 7. then, only if a `category` was provided,
        # 7a. compute the loss and add it to the output
        # 7b. update all the metrics

        # 8. finally, return the output
        return {}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {name: metric.get_metric(reset)
                for name, metric in self.metrics.items()}
