from typing import Iterable, Dict, Optional
import glob
import os
import csv

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

@DatasetReader.register("imdb")
class ImdbReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = WordTokenizer()) -> None:
        super().__init__()
        self.token_indexers = token_indexers
        self.tokenizer = tokenizer

    def text_to_instance(self, review: str, sentiment: Optional[str] = None) -> Instance:
        review_tokens = self.tokenizer.tokenize(review)
        review_field = TextField(review_tokens, self.token_indexers)

        fields = {"review": review_field}

        if sentiment is not None:
            fields["sentiment"] = LabelField(sentiment)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        print(file_path)
        for review_file in glob.glob(file_path):
            if "/pos/" in review_file:
                review_type = "pos"
            elif "/neg/" in review_file:
                review_type = "neg"
            else:
                continue
            with open(review_file) as f:
                review = f.read().strip()
                yield self.text_to_instance(review, review_type)
