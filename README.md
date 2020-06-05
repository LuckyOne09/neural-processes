# 变动部分  

model/models  
增加了Simi_repsentation 这个就是转换repsentation的FC  
修改了原来的MergeNet 现在这个只用来根据object embedding和repsentation来计算出权重  
  
新增trainer/train_simi_rep  
这里第一部分是加载general encoder 使用 xy_to_r  
第二部分是对Simi_repsentation 进行训练  
其中cnt是用来控制训练和检测的数据集的，每次只抽出一个人来进行检测，其他人用来训练  
context表示每次用的训练集    
target_x表示每次用的检测集  
target_y表示检测集的真实年龄  
x表示检测集的原始特征  
pred_age是乘以权重后的预测年龄  

# 可能的问题  
1.loss.backward(retain_graph=True) 计算图里可能有分支（我不知道这个会不会有影响，但是不加这个epoch=1就会报错）  
2.我学习率高一点训练的时候loss很快就会变成nan，可能计算相似度的地方exp还是有问题，现在一般是e^20+  
