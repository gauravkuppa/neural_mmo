# neural_mmo

conda create -n neural_mmo python=3.9

conda activate neural_mmo

pip install nmmo[rllib]

git clone https://github.com/gauravkuppa/neural_mmo.git

cd environment && pip install -e .[all]
