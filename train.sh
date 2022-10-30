model=$1
dataset=$2
gpu_ids=$3
seed=$4

for ((i=0; i<=1; i++))
do
python train.py --model ${model}\
  --dataset_mode ${dataset}\
  --gpu_ids ${gpu_ids}\
  --seed $((${seed}+${i}))\
  --num_threads 3
done