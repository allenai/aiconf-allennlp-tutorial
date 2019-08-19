FROM allennlp/allennlp:v0.8.4

RUN mkdir output/

WORKDIR /

COPY ./requirements.txt /requirements.txt

# RUN pip install -r requirements.txt

COPY ./experiments /experiments
COPY ./aiconf /aiconf

CMD ["train", "-s", "output", "--include-package", "aiconf.model_imdb", "--include-package", "aiconf.reader_imdb", "experiments/glove_bert_cased_imdb.jsonnet"]
