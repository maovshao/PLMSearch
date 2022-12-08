#Create a virtual environment 
conda create -n ss_filter python=3.8
conda activate ss_filter

#Universial
conda install -c conda-forge biopython
conda install -c conda-forge tqdm
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
#(Option for CPU only) pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
pip install matplotlib-venn
conda install -c conda-forge logzero
conda install -c conda-forge scikit-learn

#Optional
#1. To run in Jupyter Notebooks
conda install ipykernel --update-deps --force-reinstall

#2. To install tmalign from c++ source code
conda install -c conda-forge cython

#3. To run tmalign_compute by spark
pip install pyspark

#4. To generate pfam by yourself(Use pfamscan)
apt install perl-CPAN
#(or)yum install perl-CPAN
sudo cpan Moose
sudo cpan rlib
sudo apt install hmmer
#(or)yum install perl-CPAN