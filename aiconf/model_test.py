import pathlib

from allennlp.common.testing import ModelTestCase
from allennlp.data.dataset import Batch
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder

from aiconf.reader import PaperReader
from aiconf.model import PaperModel

FIXTURES_ROOT = pathlib.Path(__file__).parent / 'fixtures'


class PaperModelTest(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(FIXTURES_ROOT / 'experiment.json', FIXTURES_ROOT / 'tiny.csv')

    def test_simple_tagger_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

    # def test_forward_pass_runs_correctly(self):
    #     training_tensors = self.dataset.as_tensor_dict()
    #     output_dict = self.model(**training_tensors)
    #     output_dict = self.model.decode(output_dict)
    #     class_probs = output_dict['class_probabilities'][0].data.numpy()
    #     numpy.testing.assert_almost_equal(numpy.sum(class_probs, -1), numpy.array([1, 1, 1, 1]))
