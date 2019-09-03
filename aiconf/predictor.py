from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

from aiconf.reader import BBCReader

@Predictor.register("bbc")
class BBCPredictor(Predictor):

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        # 1. we expect that the json_dict has a "text" field and possibly
        #    a "category" field, so extract those values

        # 2. every predictor has a self._dataset_reader, so just use
        #    text_to_instance from there to return an instance
        return Instance({})
