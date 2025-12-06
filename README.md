# Modeling 5' UTR effects on translational efficiency using convolutional neural networks and human genetic variation

**Huisheng Zhu, Tami Gjorgjieva**

---

## Key Directories

| Directory | Description |
|-----------|-------------|
| `prepare_data` | Code to prepare the data we use for this project |
| `reference_inception_model` | Our primary reference model discussed in writeup |
| `variation_inception_model` | Our primary variant effect model discussed in writeup |

---

## Additional Architecture Exploration

We conducted additional architecture exploration beyond what is reported in our final project writeup. The code for experimenting with alternative architectures can be found in the following directories:

| Directory | Description |
|-----------|-------------|
| `reference_model` | Our initial model and the basis for much of `reference_inception_model/` |
| `reference_transformer_model` | Exploring transformers downstream of an inception-CNN |
| `reference_rnn_model` | Exploring RNNs downstream of an inception-CNN |
