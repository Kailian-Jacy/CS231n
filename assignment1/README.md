Details about this assignment can be found [on the course webpage](http://cs231n.github.io/), under Assignment #1 of Spring 2020`.

## 我做的时候遇到的一些坑

有关softmax的bp：
- 作业中的这个softmax层其实是softmax + cross-entropy + 一个线性全连接层组合成的，所以实现bp的一道题包含的内容很多。涉及到多种数学工具的使用。
- 而且由于整道题只有一个检查点，你中间不确定的地方一多就没法debug...

因此，建议你先查资料熟悉掌握以下这些东西，确定这些步骤是对的：
- cross-entropy的bp推导，项与项之间的偏导怎么算，写成矩阵形式的偏导怎么算，并与网上的对照。
- 矩阵点积的bp，矩阵点积怎么求导，怎么写成矩阵运算的方式求导，并对照检查。
- numpy：使用数组对高阶数组进行索引；一阶数组的两个方向升维；sum和reshape等等边边角角的问题..
- 点乘和元素积，以及在反向传播的时候层与层之间什么时候点乘什么时候元素积。

准备好再开始，不要怕麻烦。我仅仅搞对答案就用了15+小时...

在这个过程中你可以参考以下链接学习，呈现完整的方案，打好草稿、在纸上写完伪代码，确定几个检查点，查完numpy文档，再开始写代码：
- 步骤化的反向传播，合适点积何时元素积：https://zhuanlan.zhihu.com/p/37916911
- softmax求导，和一点点cross-extropy求导（交叉熵求导没太写清楚，我是另外查资料的）：https://zhuanlan.zhihu.com/p/37740860
