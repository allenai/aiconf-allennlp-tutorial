import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.token_indexers import SingleIdTokenIndexer

from aiconf.reader import PaperReader

FIXTURES_ROOT = pathlib.Path(__file__).parent / 'fixtures'


class ReaderTest(AllenNlpTestCase):
    def test_reader(self):
        token_indexers = {"tokens": SingleIdTokenIndexer()}
        reader = PaperReader(token_indexers)
        instances = reader.read(str(FIXTURES_ROOT / 'tiny.csv'))

        # Should get 10 instances
        assert len(instances) == 10

        # Each should be Science or Nature
        assert all(instance.fields["venue"].label in ["Science", "Nature"] for instance in instances)

        # First instance should be "Food Products from Plants,Nature"
        title_field = instances[0].fields["title"]
        venue_field = instances[0].fields["venue"]

        assert [token.text for token in title_field.tokens] == ["Food", "Products", "from", "Plants"]
        assert venue_field.label == "Nature"
