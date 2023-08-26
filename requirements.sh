#Create a virtual environment 
conda create -n plmsearch python=3.8
conda activate plmsearch

#conda environment
conda install -c conda-forge biopython
conda install -c conda-forge tqdm
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
#(Option for CPU only) pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
conda install -c conda-forge pandas
conda install -c conda-forge seaborn
conda install -c conda-forge logzero
conda install -c conda-forge scikit-learn
conda install -c bioconda tmalign
conda install ipykernel --update-deps --force-reinstall

#To generate pfam by yourself(Use pfamscan)
# You can follow the official steps described in "./plmsearch_data/PfamScan/README".
# The following steps are just for recommend.
sudo apt install perl-CPAN
#(or)sudo yum install perl-CPAN
sudo cpan Moose
sudo cpan rlib
sudo cpan IPC::Run
sudo apt install hmmer
#(or)sudo yum install perl-CPAN