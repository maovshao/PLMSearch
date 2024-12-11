#Create a virtual environment 
conda create -n plmsearch python=3.8
conda activate plmsearch

#conda environment
conda install -c conda-forge biopython
conda install -c conda-forge tqdm
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
# Or other pytorch versions depending on your local environment
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
conda install -c conda-forge logzero
conda install -c conda-forge scikit-learn
conda install -c bioconda tmalign
conda install ipykernel --update-deps --force-reinstall
pip install fair-esm

#To generate pfam (Use pfamscan)
# You can follow the official steps described in "./plmsearch_data/PfamScan/README".
# The following steps are just for recommend.
sudo apt install perl-CPAN
sudo apt install hmmer
sudo cpan Moose
sudo cpan rlib
sudo cpan IPC::Run