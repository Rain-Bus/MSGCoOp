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

Then, you will get the accuracy as follow:

```bash
Dataset         Base       New        H          Seeds     
------------------------------------------------------------
caltech101      98.04       95.81       96.91       3
dtd             79.32       56.64       66.09       3
eurosat         86.29       74.97       80.23       3
fgvc_aircraft   36.73       34.57       35.62       3
food101         90.74       91.62       91.18       3
imagenet        76.41       70.45       73.31       3
oxford_flowers  96.14       75.91       84.84       3
oxford_pets     95.62       97.80       96.70       3
stanford_cars   71.22       74.56       72.85       3
sun397          81.38       76.20       78.71       3
ucf101          83.52       77.05       80.16       3
------------------------------------------------------------
Average         81.55       74.48       78.10
```

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

```bash
Dataset         Base       New        H          Seeds
------------------------------------------------------------
caltech101      93.29       0.00       0.00       3
dtd             42.59       0.00       0.00       3
eurosat         45.59       0.00       0.00       3
fgvc_aircraft   19.14       0.00       0.00       3
food101         85.60       0.00       0.00       3
oxford_flowers  69.86       0.00       0.00       3
oxford_pets     87.90       0.00       0.00       3
stanford_cars   65.10       0.00       0.00       3
sun397          65.14       0.00       0.00       3
ucf101          67.51       0.00       0.00       3
------------------------------------------------------------
Average         64.17       0.00       0.00
```

> We directly using the weight from cross-domain and evaluate on RTX2080.

## Using Weight

You can reproduce our result and get our model weight on [OneDrive]();

Extract these gzip file under `MSGCoOp/MSGCoOp` and run the command above.

> NOTE: Different GPU models may lead to slightly different results.


