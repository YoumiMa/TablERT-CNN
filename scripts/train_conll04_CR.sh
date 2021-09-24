CUDA_VISIBLE_DEVICES=0 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 0 &
CUDA_VISIBLE_DEVICES=1 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 1 &
CUDA_VISIBLE_DEVICES=2 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 2 &
CUDA_VISIBLE_DEVICES=3 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 3 &
CUDA_VISIBLE_DEVICES=4 nohup python run.py train --config configs/train_conll04_scratch_CR.conf > 4 &