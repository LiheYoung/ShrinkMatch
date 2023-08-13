now=$(date +"%Y%m%d_%H%M%S")

dataset=$1
label=$2
seed=$3
port=$4

num_gpus=1

output_path=exp/shrinkmatch_${dataset}_${label}labels_seed${seed}

mkdir -p $output_path

python -m torch.distributed.launch \
    --nproc_per_node=$num_gpus \
    --master_addr=localhost \
    --master_port=$port \
    shrinkmatch.py \
    --dataset $dataset --label_per_class $label --seed $seed --DA \
    --output_path $output_path --port $port 2>&1 | tee -a $output_path/$now.log
