# PLMSearch

This is the implement of "PLMSearch: Protein language model powers accurate and fast sequence search for remote homology". By using a protein language model, PLMSearch can achieve `a sensitivity close to SOAT structure search methods` while being versatile and fast because it is `only based on sequences`.

<div align=center><img src="scientist_figures/workflow_img/main.png" width="100%" height="100%" /></div>

## Quick links

* [Webserver](#webserver)
* [Requirements](#requirements)
  * [Universial](#universial)
  * [Optional](#optional)
* [Data preparation](#data-preparation)
  * [Data](#PLMSearch-data)
  * [Protein language model](#protein-language-model)
* [Reproduce our experiments with only one file](#main)
* [Build your PLMSearch locally](#pipeline)
* [Citation](#citation)

## Webserver
<span id="webserver"></span>
Search your protein sequences in seconds using PLMSearch webserver: [issubmission.sjtu.edu.cn/PLMSearch/](https://issubmission.sjtu.edu.cn/PLMSearch/) 🚀

## Requirements
<span id="requirements"></span>

### Universial
<span id="universial"></span>

python 3.8 / biopython 1.78 / tqdm 4.64.1 / torch 1.7.1 / pandas 1.5.1 / 
seaborn 0.12.1 / matplotlib-venn 0.11.7 / logzero 1.7.0 / scikit-learn 0.24.2 /

### Optional
<span id="optional"></span>

ipykernel 6.15.2 / cython 0.29.32 / pyspark 3.3.1 / pfamscan 1.6

If you are a beginner, you can choose to follow the steps in [requirements.sh](requirements.sh).

## Data preparation
<span id="data-preparation"></span>
We provide acquirement approach of `Data` and `Protein language model`.

### Data
<span id="PLMSearch-data"></span>
We have released our prepared data **plmsearch_data** in [Webserver source](https://issubmission.sjtu.edu.cn/PLMSearch/static/download/plmsearch_data.tar.gz).
```bash
# Put it in home directory
tar zxvf plmsearch_data.tar.gz
```

### Protein language model
<span id="protein-language-model"></span>
```bash
# Download ESM-1b
cd ./plmsearch/esm/
mkdir saved_models && cd "$_"
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
# Go back to the raw directory 
cd ../../..
```

## Reproduce our experiments with only one file
<span id="main"></span>

- Reproduce our experiments with good visualization by following the steps in [main.ipynb](main.ipynb).

**Notice: the results are saved in** `scientist_figures/`.

## Build your PLMSearch locally
<span id="pipeline"></span>

- For `GPU` users, follow the example in [pipeline.ipynb](pipeline.ipynb).

- For `CPU ONLY` users, follow the example in [pipeline_cpu.ipynb](pipeline_cpu.ipynb).

**Notice: the inputs and outputs of the example are all saved in** `example/`.

## Citation
<span id="citation"></span>
```
@inproceedings{
}
```