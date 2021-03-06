#source ~/.bashrc

#installing conda
#conda create -n torch python=3.9
#source /miniconda/etc/profile.d/conda.sh
#conda activate torch

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install -r rllib_requirements.txt
pip install ray[rllib]

mkdir /home/code
cd /home/code

git clone https://github.com/gauravkuppa/neural_mmo.git

