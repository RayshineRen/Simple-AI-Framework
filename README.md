# Simple-AI-Framework
## 1.学习simpleFlow
https://github.com/PytLab/simpleflow

simpleflow

+ 四种类型结点
  + Operation
    + 输入、输出
  + Variable
    + 无输入
  + Constant
    + 值不变
  + Placeholder

前向传播图&反向传播图

梯度下降法

## 2.[如何理解TensorFlow计算图?](https://zhuanlan.zhihu.com/p/344846077)

+ **计算图模型由节点(nodes)和线(edges)组成，节点表示操作符Operator，或者称之为算子，线表示计算间的依赖，实线表示有数据传递依赖，传递的数据即张量，虚线通常可以表示控制依赖，即执行先后顺序。**
+ 计算图从本质上来说，是TensorFlow在内存中构建的程序逻辑图，计算图可以被分割成多个块，并且可以并行地运行在多个不同的cpu或gpu上，这被称为并行计算。因此，计算图可以支持大规模的神经网络。

+ TensorFlow中的计算图有三种，分别是静态计算图，动态计算图，以及Autograph。
  + 目前TensorFlow2默认采用的是动态计算图，即每使用一个算子后，该算子会被动态加入到隐含的默认计算图中立即执行得到结果。对于动态图的好处显而易见，它方便调试程序，让TensorFlow代码的表现和Python原生代码的表现一样，写起来就像写numpy一样，各种日志打印，控制流都是可以使用的。这相对于静态图来讲牺牲了些效率，因为使用动态图会有许多次Python进程和TensorFlow的C++进程之间的通信。动态计算图已经不区分计算图的定义和执行了，而是定义后立即执行，因此称之为 Eager Excution。
  + 在TensorFlow1中，采用的是静态计算图，需要先使用TensorFlow的各种算子创建计算图，然后再开启一个会话Session，显式执行计算图。静态计算图构建完成之后几乎全部在TensorFlow内核上使用C++代码执行，效率更高。此外静态图会对计算步骤进行一定的优化，剪去和结果无关的计算步骤。
  + 如果需要在TensorFlow2.0中使用静态图，可以使用@tf.function装饰器将普通Python函数转换成对应的TensorFlow计算图构建代码。运行该函数就相当于在TensorFlow1.0中用Session执行代码，使用tf.function构建静态图的方式叫做 Autograph。

+ 需要注意的是不是所有的函数都可以通过tf.function进行加速的，有的任务并不值得将函数转化为计算图形式，比如简单的矩阵乘法，然而，对于大量的计算，如对深度神经网络的优化，这一图转换能给性能带来巨大的提升。我们也把这样的图转化叫作tf.AutoGraph，在Tensorflow 2.0中，会自动地对被@tf.function装饰的函数进行AutoGraph优化。

## 3.[30天吃掉那只Tensorflow2 ](https://jackiexiao.github.io/eat_tensorflow2_in_30_days/chinese/)

## 4.SimilarWork

[leeguandong/SimilarWork: 用numpy实现tf静态图和pytorch动态图风格的深度学习框架 ](https://github.com/leeguandong/SimilarWork)

SimiliarFlow & SimilarTorch

## 5.[PyTorch 的 Autograd](https://zhuanlan.zhihu.com/p/69294347)