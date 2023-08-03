import pickle

# 重点是rb和r的区别，rb是打开2进制文件，文本文件用r
f = open("/data/hx_1/SASA/output/cfgs/kitti_models/pointrcnn_sasa/default/eval/eval_with_train/epoch_71/val/result.pkl",'rb')
carloan = pickle.load(f)
print(carloan)