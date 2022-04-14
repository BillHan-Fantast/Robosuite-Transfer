# Robosuite-Transfer
## Installation

1. Install conda environment with robosuite.yaml.
2. Install pytorch==1.8.0, torchvision compatible with the cuda driver in conda. Reference https://pytorch.org/get-started/previous-versions/.
3. Install free mujoco-py with gpu rendering (the one installed in robosuite.yaml uses cpu rendering, uninstall it first). For slurm off-screen gpu rendering, please refer to iris slack #help channel QA posted by Maximillian Du on Nov 13th, 2021.

4. After successful installation of the conda environment, srun an interactive session and run the following command in robosuite-base.
```
  $ bash scripts/local_run.sh
```
Noticing that you may have to change the PYTHONPATH in local_run.sh. For double checking, a successful gpu rendering will result in the process gpu mode as C+G.

## Copy Datasets
5. scp the offline datasets at scdt.stanford.edu:/iris/u/beining/robot-transfer/dataeset to the local NFS at Robosuite-Transfer/dataset. Noticing that these files are relatively big and will take time.

## Run robosuite-InvVAE
6. Adapt the dir paths/slurm setup in scripts/iris_vae.sh, scripts/iris_vae_hi.slrm, models/dataset.py. In models/dataset.py, it tries to load data from some default path. One can copy the data in the NFS to the local server's directory (e.g. /scr/ on Iris server). This makes dataloading much quicker.
7. Run the following in the interactive session in robosuite-InvVAE to check the paths.
```
  $ bash scripts/local_run.sh
```
8. If everythins is ok, run bash scripts/ft_launch.sh 
