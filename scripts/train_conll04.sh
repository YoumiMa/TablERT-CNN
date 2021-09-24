#$ -cwd
## Resource type F: qty 2
#$ -l rt_AF=1
## maximum run time
#$ -l h_rt=1:00:00
## output filename
#$ -N train_conll04_train_scratch_CR_8heads
#$ -o train_conll04_train_scratch_CR_8heads.out
#$ -e train_conll04_train_scratch_CR_8heads.err


source /etc/profile.d/modules.sh

module load python/3.6/3.6.12
module load intel
module load cuda/11.1/11.1.1
module load openmpi

source /home/acb11709gz/torcha/bin/activate

CUDA_VISIBLE_DEVICES=0 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 0_8heads &
CUDA_VISIBLE_DEVICES=1 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 1_8heads &
CUDA_VISIBLE_DEVICES=2 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 2_8heads &
CUDA_VISIBLE_DEVICES=3 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 3_8heads &
CUDA_VISIBLE_DEVICES=4 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 4_8heads &
wait