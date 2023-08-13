now=$(date +"%Y%m%d_%H%M%S")

data_path=$1
split=$2
num_gpus=$3
bs=$4
port=$5

output_path=exp/imagenet_$split

mkdir -p $output_path

python -m torch.distributed.launch \
    --nproc_per_node=$num_gpus \
    --master_addr=localhost \
    --master_port=$port \
    shrinkmatch.py \
    --nesterov --lambda_in 5 --lr 0.03 --epochs 400 --cos --warmup-epoch 5 --data-path $data_path --anno-percent $split --output-path $output_path \
    --c_smooth 0.9 --DA --batch-size $bs --port $port 2>&1 | tee $output_path/$now.log
