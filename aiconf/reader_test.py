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

        # Some ideas of things to test:
        # * test that there are 5 instances
        # * test that there's one of each label
        # * test that the first instance has the right values in its fields
