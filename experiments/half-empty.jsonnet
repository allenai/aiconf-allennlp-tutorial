{
    "dataset_reader": {
      "type": ???,
      "token_indexers": {
          "tokens": {
              "type": ???
          }
      }
    },
    "train_data_path": ???,
    "validation_data_path": ???,
    "model": {
      "type": ???,
      "text_field_embedder": {
        "token_embedders": {
          "tokens": {???}
        }
      },
      "encoder": {???}
    },
    "iterator": {
      "type": "bucket",
      "sorting_keys": [["text", "num_tokens"]],
      "batch_size": 2
    },
    "trainer": {
      "num_epochs": ???,
      "patience": ???,
      "cuda_device": ???,
      "grad_clipping": ???,
      "validation_metric": ???,
      "optimizer": {???}
    }
  }
