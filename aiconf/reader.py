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
        """
        1. tokenize text
        2. create a TextField for the text
        3. create a LabelField for the category (if provided)
        4. return an Instance
        """
        return Instance(fields={})

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Here our data is a csv with rows [category, text], so we want to

        1. read the csv file
        2. pass the fields to text_to_instance
        3. yield the instances
        """
        pass
