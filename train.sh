# config_file=$1
# # echo "fold num = 0"
# # python train.py -c $config_file -fn 0
# echo "fold num = 1"
# python train.py -c $config_file -fn 1
# echo "fold num = 2"
# python train.py -c $config_file -fn 2
# echo "fold num = 3"
# python train.py -c $config_file -fn 3
# echo "fold num = 4"
# python train.py -c $config_file -fn 4

config_file="effb4_exp033_stage2"
# echo "fold num = 0"
# python train_stage2.py -c $config_file -fn 0
echo "fold num = 1"
python train_stage2.py -c $config_file -fn 1
echo "fold num = 2"
python train_stage2.py -c $config_file -fn 2
echo "fold num = 3"
python train_stage2.py -c $config_file -fn 3
echo "fold num = 4"
python train_stage2.py -c $config_file -fn 4

# config_file="effb4_exp033"
# echo "fold num = 0"
# python train.py -c $config_file -fn 0
# echo "fold num = 1"
# python train.py -c $config_file -fn 1
# echo "fold num = 2"
# python train.py -c $config_file -fn 2
# echo "fold num = 3"
# python train.py -c $config_file -fn 3
# echo "fold num = 4"
# python train.py -c $config_file -fn 4

