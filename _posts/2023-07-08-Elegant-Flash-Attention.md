---
title: 优雅的Flash Attention
author: dongqi
date: 2099-07-08 15:16:00 +0800
categories: [Blogging, Tutorial]
tags: [Writing]
render_with_liquid: false
math: true
---

最近很多大模型都采用了**Flash Attention**的技术，推理速度得到了极大的提升。实际上，这只是``self-attention``针对硬件结构的一种工程优化。理论上任何使用了相同自注意力结构的模型都能使用``Flash Attention``进行训练和推理的加速。当然，这种优化和硬件的结构息息相关，[官方][flash attention implementation]目前给出的是基于``CUDA``的实现。在未来，``Flash Attention``可能会和``KV Cache``一样作为一种默认的实现集成到框架中。在这里我主要沿着论文的思路，对``Flash Attention``的优化原理进行简单说明，也可以认为是论文的阅读笔记。为了方便说明，根据论文中的伪代码，我实现了``Pytorch``版本，当然，这个版本没有任何优化。

## 背景
``Transformer``在工程上的缺陷很明显，那就是``self-attention``作为其中最主要的模块，它的时间复杂度和空间复杂度都是$$O(n^2)$$。因此，随着``sequence``长度的增加，计算会耗费极大的资源和时间。在工程算法领域，这样的复杂度几乎是不可接受的。当然，``Flash Attention``并不是为了解决这个问题，它主要从工程实现上进行了优化。在传统的计算机体系结构中，寄存器(SRAM)，高速缓存(HBM)，内存(DRAM)和磁盘的读取速度逐渐降低，但它们的容量却是逐级增加，GPU也是同理。在进行``self-attention``计算时，HBM和DRAM之间会进行频繁的数据交换(读/写)，可以简单地认为，每进行一次计算，都会先从DRAM中读取数据到HBM，计算完毕后，将数据从HBM写回到DRAM。这样子的读写是极其耗时的，GPU大部分时间都在等待数据的读写，而不是计算，效率不高。因此，``Flash Attention``从这一点出发，主要通过减少读写的操作，从而提升速度。下图是论文中给出的图，很好地说明了这些优化。
![Flash Attention](../../assets/../../assets/flash_attention_fondation.png)
_Figure 1: FlashAttention (from original paper)_

为什么说是“优雅”呢？实际上``Flash Attention``的理论相对比较简单，但是带来的效果非常显著，它很好地将数学理论和工程实践结合在了一起。

## 传统 Self-Attention 的实现
首先来看一下传统的``Self-Attention``的实现，定义$$Q, K, V \in R^{N*d}$$，其中$N$表示序列的长度，$$d$$表示``head``的维度。我们需要的计算输出为$$O \in R^{N*d}$$，因此计算过程如下所示：

$$
    S = QK^{T} \in R^{N*N}, P = softmax(S) \in R^{N*N}, O = PV \in R^{N*N}
$$

其中``softmax``是按行计算的。在标准的实现中，需要将矩阵$$S$$和$$P$$从``DRAM``加载到``HBM``中，需要使用$$O(N^2)$$的内存。通常来说，$$N >> d$$（例如，对于gpt2, N=1024, d=64）。拿``softmax``来举例，由于需要按行计算，因此需要读取整个矩阵，这是相当耗时的操作，会成为整个计算时间的瓶颈。标准实现的伪代码如下所示:
```
前提：将Q,K,V矩阵加载到HBM中
1. 将Q, K加载到HBM中，计算S，并将结果写回到HBM.
2. 从HBM中读取S，计算P，并将结果写回到HBM。
3. 从HBM中读取P，V，计算O，并将结果写回到HBM。
4. 返回O。
```

## Flash Attention 原理与实现
从传统方法的实现可以看出，$$O(n^2)$$复杂度的矩阵对HBM的重复读写是主要瓶颈，要解决这个问题需要做两件事情：
- 不需要在最后完整的矩阵上计算softmax。
- 不为反向传播存储中间阶段大的attention矩阵。
为此``flash Attention``主要提出了两种方法来解决上面的问题，``tiling``和``recomputation``。
- ``tiling``：将矩阵切割成块，分别计算attention和softmax的值，最后的值可以在多个块的结果中计算得出。
- ``recomputation``：主要是为了服务反向传播。为了不保存$$O(n^2)$$的中间矩阵，通过保存输出结果$$O$$和softmax归一化因子$$(m, l)$$，以及$$Q, K, V$$块，可以重新计算$$S, P$$。

由于重新计算，总的Flops数会增加，但是由于减少了大量的HBM读写，速度仍然比传统的实现高很多。同时，``tiling``在GEMM中是一种常用的方法。在矩阵乘法中，通常会将大的矩阵切割成小的矩阵进行计算，最后根据切割的方式将结果整合在一起。``Flash Attention``神奇的地方就在这里，通常我们算一行的``softmax``时，需要获取完整的一行才能计算，因为分母是求和计算，但是它给出的方法让我们可以通过迭代的方法分块计算``softmax``。感慨数学真是奇妙，实现相当优雅。下面我们通过数学的推理验证``tiling``和``recomputation``的正确性。

### Tiling正确性推理与前向计算
首先我们计算向量的``softmax``，假设存在一个向量$$x \in R^B$$：

$$
\begin{aligned}
m(x) &:= \max_i(x_i); \\
f(x) &:= [e^{x_1 - m(x)} ... e^{x_B - m(x)}]; \\
\ell(x) &:= \sum_i f(x)_i; \\
softmax &:= \frac{f(x)}{\ell(x)} \\
\end{aligned}
$$

假设有两个向量$$x^{(1)}, x^{(2)} \in R^B$$，我们可以把这两个拼接连在一起 $$x = [x^{(1)}, x^{(2)}]$$，并计算``softmax``，首先可以得出：

$$
m(x) := m([x^{(1)}, x^{(2)}]) = \max(m(x^{(1)}), m(x^{(2)})),\\
$$

其次我们重新构造$$f(x)$$，使得其和分块前的结果保持一致。

$$
\begin{aligned}
f(x) &:= [e^{(m(x^{(1)}) - m(x))}f(x^{(1)})  \quad e^{(m(x^{(2)}) - m(x))}f(x^{(2)})] \\
     &:= [e^{m(x^{(1)}) - m(x)}[e^{x^{(1)}_{1} - m(x^{(1)})} ... e^{x^{(1)}_{B} - m(x^{(1)})}] \quad e^{m(x^{(2)})-m(x)}[e^{x^{(2)}_{1} - m(x^{(2)})} ... e^{x^{(2)}_{B} - m(x^{(2)})}]]\\
     &:= [e^{m^{(1)}_{1} - m(x)} ... e^{m^{(2)}_{B} - m(x)}]
\end{aligned}
$$

同理，重新构造$$\ell(x)$$：

$$
\begin{aligned}
\ell(x) &:= \ell([x^{(1)}, x^{(2)}]) = e^{(m(x^{(1)}) - m(x))}\ell(x^{(1)}) + e^{(m(x^{(2)}) - m(x))}\ell(x^{(2)})\\
        &:= e^{m(x^{(1)} - m(x))}(e^{x^{(1)}_2 - m(x)} + ... + e^{x^{(1)}_B - m(x^{(1)})}) + e^{m(x^{(2)}) - m(x)}(e^{x^{(2)}_1 - m(x^{(2)})} + ... + e^{x^{(2)}_B - m(x^{(2)})})\\
        &:= e^{x^{(1)}_1 - m(x)} + ... + e^{x^{(2)}_B - m(x)}
\end{aligned}
$$

不难看出，两个函数在分块前和分块后表示的内容是一致的，因此算出来的``softmax``也是一样的。
我们首先来看一下前向计算的算法描述，能够更加直观得解释上述的公式是如何在算法中体现的。
![Flash Attention Forward Pass](../../assets/flashattention_forward_pass.png)
_Figure 2: FlashAttention Forward Pass(from original paper)_
首先根据算法1-5行的描述，需要创建几个变量，并将其分块。我们可以根据块划分的方式以及内外循环，将矩阵分为两部分。
- 第一部分：$$O = 0_{N*d} \in R^{N*d}, \ell = (0)_{N} \in R^N, m = (-\infty)_N \in R^N, Q \in R^{N*d}$$
- 第二部分：$$K, V \in R^{N*d}$$

下图展示了一次完整的外循环，每一次移动都代表了一次inner loop。为了更加直观地表示算法原理，我省略了一些不太重要的中间步骤和变量，比方说$$P$$，因为这只是在原矩阵上进行的操作，不涉及其他变量。其中着色的部分表示参与这一次计算的矩阵。
![Flash Attention Inner Loop](../../assets/flash_attention_inner_loop.gif)

其中第一部分中的矩阵全部按照第一维切分为$$T_r$$,第二部分切割为$$T_c$$。从算法描述中不难看出，计算顺序为，针对K, V的一个块，需要和整个Q做计算，这就是一轮外循环。值得注意的是第15行的计算，我们进行推导验证其的正确性。由于内循环计算的是不同的$$Q_{i}$$，而我们推导的时候是针对同一个$$Q_{i}$$的更新，因此我们假设是第$$j+1$$轮外循环。

$$
\begin{aligned}
O^{(j+1)} &= diag(\ell^{(j+1)})^{-1} (diag(\ell^{(j)})e^{m^{(j)} - m^{(j+1)}}O^{(j)} + e^{\widetilde{m} - m^{j+1}}\exp(S_{j:j+1} - \widetilde{m})V_{j:j+1})\\
&=diag(\ell^{(j+1)})^{-1} (diag(\ell^{(j)})e^{m^{(j)} - m^{(j+1)}} P_{:,:j}V_{:j} + e^{-m^{j+1}}\exp(S_{j:j+1})V_{j:j+1})\\
&=diag(\ell^{(j+1)})^{-1} (diag(\ell^{(j)})e^{m^{(j)} - m^{(j+1)}} diag(\ell^{(-j)})\exp(S_{:,:j} - m^{(j)}) V_{:j} + e^{-m^{j+1}}\exp(S_{j:j+1})V_{j:j+1})\\
&=diag(\ell^{(j+1)})^{-1} (e^{-m^{(j+1)}}\exp(S_{:,:j})V_{:j} + e^{-m^{j+1}}\exp(S_{j:j+1})V_{j:j+1})\\
&=diag(\ell^{(j+1)})^{-1} (\exp(S_{:,:j} - m^{(j+1)})V_{:j} + \exp(S_{j:j+1} - m^{(j+1)})V_{j:j+1})\\
&=diag(\ell^{(j+1)})^{-1} ( \exp(\begin{bmatrix} S_{:,:j} &  S_{j:j+1}\end{bmatrix} - m^{j+1}) )\begin{bmatrix}V_{:j}\\ V_{j:j+1} \end{bmatrix}\\
&=softmax(S_{:j+1})V_{:j+1}

\end{aligned}
$$

> 上面的推导主要参考了原论文中的附录，但是原始论文存在一个小小的typo，我在官方的github仓中提出了issue，并得到了作者的确认。[传送门](https://github.com/HazyResearch/flash-attention/issues/301)
{: .prompt-tip }

特别需要注意的是公式中每个矩阵所表示的范围，需要弄明白哪一部分表示的是新计算的block，哪一部分表示的是上一步的结果。

按块计算``softmax``是``flash attention``的核心思想，因此，我们进一步了解它是怎么发生的，这对理解整个算法非常重要。我们观察矩阵``S``的一行，并遍历外循环，如下图所示。

### Recomputation正确性推理与反向计算
首先看一下传统attention计算中的反向计算，截图取自论文。
![Standard Attention Backward Pass](../../assets/standard_attention_backward_pass.png)
_Figure 3: Standard Attention Backward Pass(from original paper)_
可知传统attention反向计算输入为5个矩阵，分别是$$Q, K, V, dO \in R^{N*d}, P \in R^{N*N}$$，而在flash attention的前向传播算法中并没保存$$P \in R^{N*N}$$，显然如果要进行反向计算，必须要进行一些额外的运算来取代这个矩阵，以达到相同的功能。再来看一下flash attention的方向传播算法:
![Flash Attention Backward Pass](../../assets/flashattention_backward_pass.png)
_Figure 4: Flash Attention Backward Pass_
算法的输入为7个矩阵，分别是$$Q, K, V, O, dO \in R^{N*d}$$, 以及前向传播计算返回的两个向量$$\ell, m \in R^{N}$$。在上图中的第11-13行，使用这两个向量重新计算了P，也就是标题中的所说的``Recomputation``。虽然重复计算增加了计算的总数，但是作者通过实验表明，这里不会成为算法的瓶颈，计算相比传统算法，依然得到了很大的提升。



### IO复杂度分析





## Reference
[Flash Attention Paper][Flash Attention Paper]

[Flash Attention Implementation][flash attention implementation]



[Flash Attention Paper]: https://arxiv.org/pdf/2205.14135.pdf
[flash attention implementation]: https://github.com/HazyResearch/flash-attention
