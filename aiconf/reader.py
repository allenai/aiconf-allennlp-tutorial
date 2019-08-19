from typing import Iterable, Dict, Optional
import gzip
import csv

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

@DatasetReader.register("paper")
class PaperReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer],
                 tokenizer: Tokenizer = WordTokenizer()) -> None:
        super().__init__()
        self.token_indexers = token_indexers
        self.tokenizer = tokenizer

    def text_to_instance(self, title: str, venue: Optional[str] = None) -> Instance:
        title_tokens = self.tokenizer.tokenize(title)
        title_field = TextField(title_tokens, self.token_indexers)

        fields = {"title": title_field}

        if venue is not None:
            fields["venue"] = LabelField(venue)

        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        # handle both gzipped and non-gzipped files
        my_open = gzip.open if file_path.endswith(".gz") else open

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with my_open(file_path, 'rt') as f:
            reader = csv.reader(f)
            for title, venue in reader:
                yield self.text_to_instance(title, venue)
