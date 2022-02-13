for seed in 2019 2020 2021
do
    CUDA_VISIBLE_DEVICES=$1 python train.py --config_path multiwoz${2}/${3}configs/multiwoz_all_context.json --seed ${seed}
    CUDA_VISIBLE_DEVICES=$1 python test.py --config_path multiwoz${2}/${3}configs/multiwoz_all_context.json --seed ${seed}
done