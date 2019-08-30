from typing import Iterable, Dict, Optional
import gzip
import csv

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

@DatasetReader.register("bbc")
class BBCReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = WordTokenizer()) -> None:
        super().__init__()
        self.token_indexers = token_indexers
        self.tokenizer = tokenizer

    def text_to_instance(self, text: str, category: Optional[str] = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.token_indexers)

        fields = {"text": text_field}

        if category is not None:
            fields["category"] = LabelField(category)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'rt') as f:
            reader = csv.reader(f)
            for category, text in reader:
                yield self.text_to_instance(text, category)
