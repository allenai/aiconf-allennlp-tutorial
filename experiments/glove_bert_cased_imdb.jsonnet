local model = "bert-base-cased";

{
    "dataset_reader": {
      "type": "imdb",
      "tokenizer": {
          "word_splitter": "bert-basic"
      },
      "token_indexers": {
          "tokens": {
              "type": "bert-pretrained",
              "pretrained_model": model
          }
      }
    },
    "train_data_path": "data/aclImdb/train/*/*.txt",
    "validation_data_path": "data/aclImdb/test/*/*.txt",
    "model": {
      "type": "imdb",
      "text_field_embedder": {
        "allow_unmatched_keys": true,
        "token_embedders": {
          "tokens": {
            "type": "bert-pretrained",
            "pretrained_model": model,
            "top_layer_only": true,
            "requires_grad": false
          }
        }
      },
      "encoder": {
        "type": "bert_pooler",
        "pretrained_model": model,
        "requires_grad": true
      }
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["review", "num_tokens"]],
      "batch_size": 10
    },
    "trainer": {
      "num_epochs": 50,
      "patience": 10,
      "cuda_device": 0,
      "grad_clipping": 5.0,
      "validation_metric": "+accuracy",
      "optimizer": {
        "type": "adagrad"
      }
    }
  }
