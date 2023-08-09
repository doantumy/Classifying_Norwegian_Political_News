# Classifying Norwegian Political News

## Data
Link to data: 

```https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-4/```

## Model training

To train model please use below command and replace the argument values corresponding to datasets. Details about parameters are available in the paper:

[1] Full fine-tuning model:

```
# Example training code

python news_classifier.py \
                --batchsize=64 \
                --traindataset=./data/balanced_train.tsv \
                --testdataset=./data/test.tsv \
                --epoch=15 --language=nor \
                --learningrate=0.0001 \
                --maxlength=512 \
                --modelname=nb-bert \
                --modelpath=NbAiLab/nb-bert-base \
                --numlabels=2 \
                --warmupstep=1000 \
                --weightdecay=0.01 \
                --projectname=specify your wandb project folder here

```

[2] Adapter fine-tuning model:
```
# Example training code

python news_classifier_adapter.py \
                --batchsize=64 \
                --traindataset=./data/balanced_train.tsv \
                --testdataset=./data/test.tsv \
                --epoch=15 \
                --language=nor \
                --learningrate=0.001 \
                --maxlength=512 \
                --modelname=nb-bert \
                --modelpath=NbAiLab/nb-bert-base \
                --numlabels=2 \
                --warmupstep=1000 \
                --weightdecay=0.01 \
                --projectname='specify your wandb project folder here'
```

## To cite the paper
```
TBA
```
