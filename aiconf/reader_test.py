import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import SingleIdTokenIndexer

from aiconf.reader import BBCReader

FIXTURES_ROOT = pathlib.Path(__file__).parent / 'fixtures'


class ReaderTest(AllenNlpTestCase):
    def test_reader(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        reader = BBCReader(token_indexers)
        instances = reader.read(str(FIXTURES_ROOT / 'tiny.csv'))

        # Should get 5 instances
        assert len(instances) == 5

        # should have one of each label
        labels = {instance.fields["category"].label for instance in instances}
        assert labels == {"business", "sport", "entertainment", "politics", "tech"}

        # First instance should be "computer grid"
        text_field = instances[0].fields["text"]
        category_field = instances[0].fields["category"]

        assert [token.text for token in text_field.tokens] == [
            "Computer", "grid", "to", "help", "the", "world",
            "Your", "computer", "can", "now", "help", "solve", "the", "worlds", "most", "difficult", "health", "and", "social", "problems"
        ]
        assert category_field.label == "tech"
