%这段matlab脚本的作用是将高光谱图像数据（近邻像素处理后的）训练集组装成caffe认识的hdf5格式的文件
h5create('test.hd5','/data',[1 1 1584 1050],'Datatype','single');
h5create('test.hd5','/label',[1 1 1 1050],'Datatype','single');
test_data = reshape(DataTe, [1 1 1584 1050]);
test_label = reshape(CIdTe, [1 1 1 1050]);
h5write('test.hd5','/data',single(test_data));
h5write('test.hd5','/label',single(test_label));
