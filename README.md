## This is the latest version of the code that was used for our EMNLP 2018 paper

## This version includes
* PyTorch 0.4
* MultiGPU training of model
* Joint training of both non-autoregressive model and length prediction module at the same time
* WMT'14 datasets and pretrained non-autoregressive models

==================================

Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement
==================================
PyTorch implementation of the models described in the paper [Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement](https://arxiv.org/abs/1802.06901 "Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement").

We present code for training and decoding both autoregressive and non-autoregressive models, as well as preprocessed datasets and pretrained models.

Dependencies
------------------
### Python
* Python 3.6
* PyTorch 0.4
* Numpy
* NLTK
* torchtext ( you need to install my modified multigpu version https://github.com/mansimov/pytorch_text_multigpu )
* torchvision

### GPU
* CUDA (we recommend using the latest version. The version 8.0 was used in all our experiments.)

### Related code
* For preprocessing, we used the scripts from [Moses](https://github.com/moses-smt/mosesdecoder "Moses") and [Subword-NMT](https://github.com/rsennrich/subword-nmt "Subword-NMT").
* This code is based on [NA-NMT](https://github.com/MultiPath/NA-NMT "NA-NMT").

Downloading Datasets & Pre-trained Models
------------------
The original translation corpora can be downloaded from ([IWLST'16 En-De](https://wit3.fbk.eu/), [WMT'16 En-Ro](http://www.statmt.org/wmt16/translation-task.html), [WMT'15 En-De](http://www.statmt.org/wmt15/translation-task.html), [MS COCO](http://cocodataset.org/#home)). For the preprocessed corpora and pre-trained models, see below.

| | Dataset | Model |
| -------------      | --- | -------------  |
| IWSLT'16 En-De     | [Data](https://drive.google.com/file/d/1m7dZqEXHWPYcre6xxsFwFLrb9CRCZGmn/view?usp=sharing) | [Models](https://drive.google.com/open?id=1N8tfU5ttnov2jWk3-PHVMJClQA0pKXoN) |
| WMT'16 En-Ro       | [Data](https://drive.google.com/file/d/1YrAwCEuktG-iDVxtEW-FE72uFTLc5QMl/view?usp=sharing) | [Models](https://drive.google.com/open?id=1qHSkrmTgj5c4U54zJZomdXQ_YUbhhfVi) |
| WMT'15 En-De       | [Data](https://drive.google.com/file/d/1Q5-54S34HgC36IxJEZLwduKKW_EXZHWb/view?usp=sharing) | [Models](https://drive.google.com/open?id=1TJobn-RNxMDNLBqgglAmhA5vV1kpCebf) |
| WMT'14 En-De (**new**)      | [Data](https://drive.google.com/file/d/1t7w0dmURRkXIbmzzlIUhrffw8eYctsIT/view?usp=sharing) | [Models](https://drive.google.com/file/d/1Wmzm5V_jxffT3yfZ7xVt7WX17ldNzpUW/view?usp=sharing) |
| MS COCO            | [Data](https://drive.google.com/open?id=10RJbEb71CQZzaPtvS__KS50Fi5SrHHTN) | [Models](https://drive.google.com/open?id=1hqT9Hf8nGlWP9pqyAg4KRDR1QfDCvW4z) |

Before you run the code
------------------
Set correct path to data in `data_path()` function located in [`data.py`](https://github.com/jasonleeinf/non-auto-decoding/blob/96f7765399133c79ad4d23768dd530ee3eb07990/data.py#L44):

Loading & Decoding from Pre-trained Models
------------------
1. For `vocab_size`, use `60000` for WMT'14 En-De, `60000` for WMT'15 En-De, `40000` for the other translation datasets and `10000` for MS COCO.
2. For `params`, use `big` for WMT'15 En-De and `small` for the other translation datasets.

#### Autoregressive
```bash
$ python run.py --dataset <dataset> --vocab_size <vocab_size> --ffw_block highway --params <params> --lr_schedule anneal --mode test --debug --load_from <checkpoint>
```

#### Non-autoregressive
```bash
$ python run.py --dataset <dataset> --vocab_size <vocab_size> --ffw_block highway --params <params> --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --mode test --remove_repeats --debug --trg_len_option predict --use_predicted_trg_len --load_from <checkpoint>
```

For adaptive decoding, add the flag `--adaptive_decoding jaccard` to the above.

Training New Models
------------------

#### Autoregressive
```bash
$ python run.py --dataset <dataset> --vocab_size <vocab_size> --ffw_block highway --params <params> --lr_schedule anneal
```

#### Non-autoregressive
```bash
$ python run.py --dataset <dataset> --vocab_size <vocab_size> --ffw_block highway --params <params> --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --denoising_prob --layerwise_denoising_weight --use_distillation
```

Training the Length Prediction Model
------------------
1. Take a checkpoint pre-trained non-autoregressive model
2. Resume training using these in addition to the same flags used in step 1: `--load_from <checkpoint> --resume --finetune_trg_len --trg_len_option predict`

MS COCO dataset
------------------

* Run pre-trained autoregressive model

```
python run.py --dataset mscoco --params big --load_vocab --mode test --n_layers 4 --ffw_block highway --debug --load_from mscoco_models_final/ar_model --batch_size 1024
```

* Run pre-trained non-autoregressive model

```
python run.py --dataset mscoco --params big --use_argmax --load_vocab --mode test --n_layers 4 --fast --ffw_block highway --debug --trg_len_option predict --use_predicted_trg_len --load_from mscoco_models_final/nar_model --batch_size 1024
```

* Train new autoregressive model

```
python run.py --dataset mscoco --params big --batch_size 1024 --load_vocab --eval_every 1000 --drop_ratio 0.5 --lr_schedule transformer --n_layers 4
```

* Train new non-autoregressive model

```
python run.py --dataset mscoco --params big --use_argmax --batch_size 1024 --load_vocab --eval_every 1000 --drop_ratio 0.5 --lr_schedule transformer --n_layers 4 --fast --use_distillation --ffw_block highway --denoising_prob 0.5 --layerwise_denoising_weight --load_encoder_from mscoco_models_final/ar_model
```

After training it, train the length predictor (set correct path in `load_from` argument)

```
python run.py --dataset mscoco --params big --use_argmax --batch_size 1024 --load_vocab --mode train --n_layers 4 --fast --ffw_block highway --eval_every 1000 --drop_ratio 0.5 --drop_len_pred 0.0 --lr_schedule anneal --anneal_steps 100000 --use_distillation --load_from mscoco_models/new_nar_model --trg_len_option predict --finetune_trg_len --max_offset 20
```

Citation
------------------
If you find the resources in this repository useful, please consider citing:
```
@article{Lee:18,
  author    = {Jason Lee and Elman Mansimov and Kyunghyun Cho},
  title     = {Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement},
  year      = {2018},
  journal   = {arXiv preprint arXiv:1802.06901},
}
```
