```
cd ~/FCN-pytorch
python3 python/kitti_utils.py -d "/home/yu/DataSet/data_semantics" -nc 3 --width 1024 --height 320 --resize --calculate_mean

```

python3 python/train_kitti.py -d "/home/yu/DataSet/data_semantics" --model fcns --batch_size 16 -nc 3 --width 1024 --height 320