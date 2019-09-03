from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

from aiconf.reader import BBCReader

@Predictor.register("bbc")
class BBCPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        text = json_dict["text"]
        category = json_dict.get("category")
        return self._dataset_reader.text_to_instance(text, category)
