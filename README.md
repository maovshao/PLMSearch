# PLMSearch

This is the implement of "PLMSearch: Protein language model powers accurate and fast sequence search for remote homology". By using a protein language model, PLMSearch can achieve `a sensitivity close to SOAT structure search methods` while being versatile and fast because it is `only based on sequences`.

<div align=center><img src="scientist_figures/workflow_img/framework.png" width="100%" height="100%" /></div>

## Quick links

* [Webserver](#webserver)
* [Requirements](#requirements)
* [Data preparation](#data-preparation)
* [Reproduce all our experiments with only one file](#main)
* [Run PLMSearch locally](#pipeline)
* [Citation](#citation)

## Webserver
<span id="webserver"></span>

PLMSearch web server : [dmiip.sjtu.edu.cn/PLMSearch](https://dmiip.sjtu.edu.cn/PLMSearch/) 🚀

PLMAlign  web server : [dmiip.sjtu.edu.cn/PLMAlign](https://dmiip.sjtu.edu.cn/PLMAlign/) :airplane:

PLMAlign source code : [github.com/maovshao/PLMAlign](https://github.com/maovshao/PLMAlign/) :helicopter:

## Requirements
<span id="requirements"></span>

Follow the steps in [requirements.sh](requirements.sh)

## Data preparation
<span id="data-preparation"></span>

We have released our experiment data in [plmsearch_data](https://dmiip.sjtu.edu.cn/PLMSearch/static/download/plmsearch_data.tar.gz).
```bash
# Include experiment data, PLMSearch model, ESM-1b model, etc.
wget https://dmiip.sjtu.edu.cn/PLMSearch/static/download/plmsearch_data.tar.gz
tar zxvf plmsearch_data.tar.gz
```

## Reproduce all our experiments with only one file
<span id="main"></span>

- Reproduce all our experiments with good visualization by following the steps in [main.ipynb](main.ipynb)

**Notice: Detailed results are saved in** `scientist_figures/`.

## Run PLMSearch locally
<span id="pipeline"></span>

- Run PLMSearch locally by following the example in [pipeline.ipynb](pipeline.ipynb)

**Notice: the inputs and outputs of the example are saved in** `example/`.

## Citation
<span id="citation"></span>
Liu, W., Wang, Z., You, R. et al. PLMSearch: Protein language model powers accurate and fast sequence search for remote homology. Nat Commun 15, 2775 (2024). https://doi.org/10.1038/s41467-024-46808-5