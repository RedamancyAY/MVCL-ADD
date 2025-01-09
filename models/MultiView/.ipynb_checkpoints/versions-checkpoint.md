- 28, 29: 使用self.norm_feat
- 30， 31: 使用self.norm_feat，去除clip loss中的batch norm

以上效果都不好，低于90，但是又是正常需要的，因此保持以上选择


- 34， 35: 冻结wavlm的1D CNN
- 36，37： 冻结wavlm的1D CNN，和resnet的stage1

很有效果，第一个epoch的auc达92左右，但是3个epoch之后，就开始下降。


- 38，39：添加1D和2D的分类损失
- 40， 41：同时添加1D和2D的对比损失

又提了一点，现在是93左右，跑5个epoch，可以到94


- 42, 43：在前两个epoch只使用clip loss
    - 42， e5, 92.8
    - 43， e5, 93.5/12.44
- 44, 45：在前1个epoch只使用clip loss
  - 44, e6, 93.5/13.7
    - 45, e1, 91.5

----
8.11

- 3, 4, ：在前2个epoch只使用clip loss + mse loss
    - 后面就过拟合了
- 9, 10 ：使用所有损失，使用改进的squeeze module
    - v9: epoch 1 94.8/11.3，但是后面就过拟合了

- 15,16：使用所有损失，使用改进的squeeze module with multi-head attention
      -v 15,开始小于90，但是后面会上来一点
      -v 16, epoch 1 94.1/11.8，但是后面就过拟合了


########################################################################################
不管是squeeze module，还是expand module，不使用multi-head，性能会更好一点
###################################################################################################

- 70, 71 调整mse loss的实现
    - 只有91，直接过拟合
 
- 72, 73, 调整mse loss的实现，在前2个epoch只使用clip loss + mse loss
- 74，75，调整mse loss的实现，在前2个epoch只使用clip loss + mse loss，后面删除这两个损失
   - e4, 94.6/12.7
 
