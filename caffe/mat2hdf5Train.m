%这段matlab脚本的作用是将高光谱图像数据（近邻像素处理后的）训练集组装成caffe认识的hdf5格式的文件
h5create('train.hd5','/data',[1 1 1584 4145],'Datatype','single');
h5create('train.hd5','/label',[1 1 1 4145],'Datatype','single');
train_data = reshape(DataTr, [1 1 1584 4145]);
train_label = reshape(CIdTr, [1 1 1 4145]);
h5write('train.hd5','/data',single(train_data));
h5write('train.hd5','/label',single(train_label));
