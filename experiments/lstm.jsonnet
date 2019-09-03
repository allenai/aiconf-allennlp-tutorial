{
    "dataset_reader": {
      "type": "bbc",
      "token_indexers": {
          "tokens": {
              "type": "single_id"
          }
      }
    },
    "train_data_path": "data/bbc-train.csv",
    "validation_data_path": "data/bbc-validate.csv",
    "model": {
      "type": "bbc",
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {
            "type": "embedding",
            "embedding_dim": 100,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
            "trainable": false
          }
        }
      },
      "encoder": {
        "type": "lstm",
        "input_size": 100,
        "hidden_size": 25,
        "num_layers": 1,
        "bidirectional": true
      }
      //"dropout": 0.8
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["text", "num_tokens"]],
      "batch_size": 10
    },
    "trainer": {
      "num_epochs": 100,
      "patience": 10,
      "cuda_device": -1,
      "grad_clipping": 5.0,
      "validation_metric": "+acc1",
      "optimizer": {
        "type": "adagrad"
      }
    }
  }
