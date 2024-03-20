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

+ SimilarFlow
  + Operation重载的__mul__有区别，similarflow是matmul，相当于@，而simpleflow是element-wise的乘法。
  + 注意graph.py中的导入方式，否则会出现循环导入的问题。
  + python -m exam_simiflow.test_ffd测试成功。
+ SimilarTorch

## 5.PyTorch原理

[PyTorch 的 Autograd](https://zhuanlan.zhihu.com/p/69294347)

[一文搞懂 PyTorch 内部机制](https://zhuanlan.zhihu.com/p/338256656)

+ PyTorch的"三要素"：布局(layout)，设备(device)和数据类型(dtype)
  + 设备类型(The device) 设备类型描述了tensor的到底存储在哪里，比如在CPU内存上还是在NVIDIA GPU显存上，在AMD GPU(hip)上还是在TPU(xla)上。不同设备的特征是它们有自己的存储分配器(allocator)，不同设备的分配器不能混用。
  + 内存布局(The layout) 描述了我们如何解释这些物理内存。常见的布局是基于步长的tensor(strided tensor)。稀疏tensor有不同的内存布局，通常包含一对tensors，一个用来存储索引，一个用来存储数据；MKL-DNN tensors 可能有更加不寻常的布局，比如块布局(blocked layout)，这种布局难以被简单的步长(strides)表达。
  + 数据类型(The dtype) 数据类型描述tensor中的每个元素如何被存储的，他们可能是浮点型或者整形，或者量子整形。
+ Tensor 是PyTorch的核心数据结构。
  + 可以认为tensor包含了数据和元数据(metadata)，元数据用来描述tensor的大小、其包含内部数据的类型、存储的位置(CPU内存或是CUDA显存)
  + Stride。Tensor是一个数学概念。当用计算机表示数学概念的时候，通常我们需要定义一种物理存储方式。最常见的表示方式是将Tensor中的每个元素按照次序连续的在内存中铺开(这是术语contiguous的来历)，将每一行写到相应内存位置里。步长用来将逻辑地址转化到物理内存的地址。当我们根据下标索引查找tensor中的任意元素时，将某维度的下标索引和对应的步长相乘，然后将所有维度乘积相加就可以了。
  + 要取得tensor上的视图，我们得对tensor的的逻辑概念和tensor底层的物理数据(称为存储 storage)进行解耦。一个存储可能对应多个tensor。存储定义了tensor的数据类型和物理大小，而每个tensor记录了自己的大小(size)，步长(stride)和偏移(offset)，这些元素定义了该tensor如何对存储进行逻辑解释。
  + tensor上的操作(operations)是如何实现的？当你调用`torch.mm`的时候，会产生两种分派(dispatch)：第一种分派基于设备类型(device type)和tensor的布局(layout of a tensor)，例如这个tensor是CPU tensor还是CUDA tensor；或者，这个tensor是基于步长的(strided) tensor 还是稀疏tensor。第二种分派基于tensor的数据类型(dtype)。这种依赖可以通过简单的`switch`语句解决。

[PyTorch结构、架构分析_pytorch架构](https://blog.csdn.net/qq_28726979/article/details/120690343)

[PyTorch JIT](https://zhuanlan.zhihu.com/p/370455320)

+ **TorchScript**
  + [louis-she/torchscript-demos: A brief of TorchScript by MNIST](https://github.com/louis-she/torchscript-demos)

## 6.Tensorflow-internals

## 7.[PyTorch – Internal Architecture Tour](https://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour/)

