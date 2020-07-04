# Cost-net
A network for produce stereo cost-volume.
This work is trying to re-implement the closed-source project proposed by <A Deep Visual Correspondence Embedding Model for Stereo Matching Costs> with pytorch.

Contact: [wsywf@bupt.edu.cn](mailto:wsywf@bupt.edu.cn). Any questions or discussions are welcomed!  

## Usage

If you want to train the model,you can follow two ways to pretreat the data.  
1.You can just use the image like KITTI as the source data, use the data_crop.py to load dataset and prepare the data with training,but this way might cost more time in each epoch.

2.You can also prepare the data before you train with the use of the file: data_prepare/pair_load.py,and you can choose the data_load_prepared.py as your dataloder.

./data_crop.py.py ------------------------- load the un-prepared data.

./data_load_prepared.py ------------------- load the prepared data..

./train_cost.py --------------------------- Train the COST-NET.

./cost_Net_new.py ------------------------- The Cost-net model.

./test.py --------------------------------- To get the cost-volume with the trained model.

If you want to use the params to post-procedure with c++,you can set the save_path in save2ctype.cpp and then use command     g++ -fPIC -shared -o libsave.so save2ctype.cpp  to build a dynamic link library and use it to get a c_type-cost-volume.
