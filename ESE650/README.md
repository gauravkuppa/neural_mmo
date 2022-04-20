# 1. First setup a docker image from the provided docker file eg docker build --tag neuralmmo:torch3.9 .

# 2. Create a container eg docker run -it --gpus all --net host --ipc host --name nmmo1 neuralmmo:torch3.9 bash

# 3. In the terninal source bashrtc
# 4. Create a conda environemnt with python=3.9
# 5. activate conda environment and run the startup.sh script
# 6. go to environment in /home/code  where neural_mmo is cloned and pip install -e . to install nmmo environment
# 7. install wandb openskill and any other necessary packages to run the code in ESE650/main.py 