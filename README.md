# CS231n

CV 入门课 CS231n assignment.

## Source 

- Lecture: https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk

- Notes: https://changeable-foxtrot-097.notion.site/CS231n-Convolutional-Neural-Networks-for-Visual-Recognition-7e993d63a1f645d7a4b30b963e1e284a

- Assignment: https://cs231n.github.io

## Suggestions

..To be updated.

### Assignment1

Part1: KNN

写了快两天，时间主要花在：
- numpy查资料和解决corner case
- debug。尤其是在不知道结果是否正确的部分。

建议：
- 打草稿、列出足够的细节，然后动手写代码。在写之前、构思解法的时候先熟悉函数。
- 检查入参。至少在使用的时候先熟悉入参。
- 在前期解决的时候设计debug点。
- numpy降低错误率的重要手段是关注shape。时刻print shape来防止错误浪费时间。

Part2: svm

改进：
- 通过数学演算的方式详细推导结果。
- 尝试在开始推导前熟悉入参，将草稿的入参和函数提供的保持一致。
- 跟踪shape
- 提前查找numpy函数，保证思维顺畅。

提高：
- 没有习惯这种“广义数学”式的演算推导。将numpy的一些函数引入正常演算，却无法灵敏、及时地明确自己需要得到的结果长啥样、怎么获得。总而言之就是还不习惯。
- 未关注类型。numpy在计算时，如果出现整形类型，一定要投以高度重视。