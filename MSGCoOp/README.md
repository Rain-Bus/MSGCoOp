# How to Run

## GPU memory needed

All the experiments is able to run on a single graphic card. If you want to reproduce our result, you can train on vGPU-32G and vGPU-48G provided by [AutoDL](https://www.autodl.com/docs/gpu_perf/).


## How to Install

This code is built on top of the toolbox [Dassl.ProGrad.pytorch](https://github.com/BeierZhu/Prompt-align/tree/main/Dassl.ProGrad.pytorch). You can prepare the environment as follows:

```
# Create a conda environment
conda create -n msgcoop python=3.12

# Activate the environment
conda activate msgcoop

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `MSGCoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.


## Generalization From Base to New Classes

You will need `base2new_train.sh`, `base2new_test.sh`, and `base2new_all.sh`. The scripts with the prefix `base2new_train` train a model on base classes while the ones with the prefix `base2new_test` evaluate the trained model on new classes. Both kinds of scripts have three input argument, i.e., `TRAINER SG_WEIGHT DIV_WEIGHT`.

You can run base to new on all datasets as follow:

```bash
bash scripts/base2new_all.sh MSGCoOp 8.0 1.0
```

When the evaluation is done, you can use `extract_acc.py` (replace the `root_dir` in the `main` function to your output dir) to automatically calculate the average results. For instance, after you finish the trainning using the aforementioned commands, you would get

```
output
└── base2new
    ├── test_new
    │   ├── caltech101
    │   │   └── shots_16_8.0
    │   │       └── MSGCoOp
    │   │           └── vit_b16_ep100_ctxv1
    │   │               ├── seed1
    │   │               ├── seed2
    │   │               └── seed3
    │   ├── dtd
    │   │   └── shots_16_8.0
    │   │       └── MSGCoOp
    │   │           └── vit_b16_ep100_ctxv1
    │   │               ├── seed1
    │   │               ├── seed2
    │   │               └── seed3
    │   ├── ...
    └── train_base
        ├── caltech101
        │   └── shots_16_8.0
        │       └── MSGCoOp
        │           └── vit_b16_ep100_ctxv1
        │               ├── seed1
        │               ├── seed2
        │               └── seed3
        ├── dtd
        │   └── shots_16_8.0
        │       └── MSGCoOp
        │           └── vit_b16_ep100_ctxv1
        │               ├── seed1
        │               ├── seed2
        │               └── seed3
        ├── ...
```

Then, you will get the avarage accuracy.

> We train and evaluate our model on vGPU-32G.

## Generalization For Cross Domain

Fisrt, you need train on all classes over ImageNet:

```bash
bash scripts/xd_train.sh MSGCoOp 8.0 1.0
```

Then you can evaluate the performance on other ImageNet variants by run:

```bash
bash scripts/xdo_test.sh MSGCoOp 8.0 1.0
```

And you will get the `output_xdo` after script finish. You can get the accuracy by `extract_acc.py` (need modify the `root_dir` to `output_xdo` ).

> We train and evaluate our model on vGPU-48G.

## Generalization For Cross Dataset

Directly run follow command for get the result of cross dataset:

```bash
bash scripts/xda_test.sh MSGCoOp 8.0 1.0
```

You also can achive the result by `extract_acc.py`:

> We directly using the weight from cross-domain and evaluate on RTX2080.

# Use Weight

We provide our weight for reproduce. You can download it with [OneDrive](https://1drv.ms/f/c/b31f6836d06839db/Ets5aNA2aB8ggLOxawAAAAABViDG_EDvsE1cPq4PxWixog?e=nka3O0). Only need extract these weight to `MSGCoOp/MSGCoOp`, and run the shell scripts of each benchmark expriments. Then you can easily got the result.

> NOTE: Different GPUs may cause slightly different result.
