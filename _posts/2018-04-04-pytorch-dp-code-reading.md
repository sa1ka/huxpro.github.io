---
layout: post
title:  DataParallel pytorch 代码阅读
tags:
  - multigpu
  - pytorch
  - data parallel
---

# 前言

最近在做大规模深度神经网络并行的事情，因此模型的多卡并行训练成了必须的事情．并行化可在在模型或者数据两个层面上进行，相比模型并行，数据并行更容易实现而且额外的开销的更小，模型并行一般在模型十分巨大的情况下才会使用（比如一个超大的神经网络，无法存储在一张 GPU 内）．

虽然 Pytorch 在多机多卡方面做的并不如 MXNet，但是其提供的 DataParallel \(DP\) 和 DistributedDataParallel \(DDP\) 还是值得我们这些初学者好好研究的．虽然 DP 和 DDP 在功能上的很相似（一个负责单机多卡，一个负责多机多卡），但两个框架的内部实现的逻辑差别是很大的．

本文的目的在于理清这两个框架的主要实现逻辑，不会深入到具体每行代码中的细节．

# 源码阅读

## DataParallel

DP 的实现比较简单，代码量并不多，而且基本都是 python 代码，因此比较适合阅读．DP的主要逻辑如下:

```
def forward(self, *inputs, **kwargs):
    if not self.device_ids:
        return self.module(*inputs, **kwargs)
    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    if len(self.device_ids) == 1:
        return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    return self.gather(outputs, self.output_device)
```

forward 阶段，首先 scatter 把数据切分成多分（默认在dim=0上切分），然后 replicate 将模型拷贝多份，分散到不同的卡上．parallel_apply 负责多卡的并行计算，使用多线程实现．计算后的结果最后再由 gather 合并到一起，交给之后的 criterion 计算 loss.

backward 阶段，所有卡上模型的梯度会累加到原始的模型中，但这一步值得我们仔细深究．首先 forward 中用到的所有操作 (scatter, replicate, gather）都是可以计算 backward 的.

其中 scatter 与 gather, 实际上调用了 `nn/parallel/_functions.py` 中的 `Gather` 与 `Scatter`，`Gather` 的源码如下:

```
class Gather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        assert all(map(lambda i: i.is_cuda, inputs))
        ctx.target_device = target_device
        ctx.dim = dim
        ctx.input_gpus = tuple(map(lambda i: i.get_device(), inputs))
        ctx.input_sizes = tuple(map(lambda i: i.size(ctx.dim), inputs))
        return comm.gather(inputs, ctx.dim, ctx.target_device)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None) + Scatter.apply(ctx.input_gpus, ctx.input_sizes, ctx.dim, grad_output)
```

可以看到，`Gather`与`Scatter`互为逆操作，而且再底层就是调用 pytorch 的 `comm`进行GPU通讯了．

而 replicate 实际调用的是`nn/parallel/_functions.py`中的`Broadcast`，与`Broadcast`互为逆操作的是`ReduceAddCoalesced`．所以在 forward 阶段，`Broadcast`负责将模型分到多个 GPU 上；backward 阶段，`ReduceAddCoalesced` 负责将多个模型的梯度汇聚到某一张卡上．

parallel_apply 才是模型真正的并行计算部分，这部分是用多线程实现的，每个线程负责各自的 GPU，而且这些线程在计算结束后，就释放了. 所以在 DP 中，虽然利用了多卡，但这些卡是由一个进程管理的，只是在计算的时候使用了多个线程，但因为 GIL 的缘故，多线程也只能利用单核. 这对于 CPU 资源利用比较大的模型有天然的限制．Pytorch 的官方文档也说了 DP 不适合使用在 RNN 中．在实际使用中，使用 DP 来训练 LSTM 语言模型反而会拖慢训练速度．

### 一点疑惑

正如之前所说，forward 时的多线程在计算完毕后就释放了，那么 backward 的时候是否还需要多线程呢？这部分涉及到 Pytorch 计算 backward 的细节了．我们知道，Pytorch 会把 forward 阶段的计算图存下来供 backward 使用，而 backward 的计算顺序实际上是 forward 计算图的一个反向拓扑排序．现在的问题在于，我们会有多张这样的计算图，各个计算图的 backward 是独立的，那么它们也是并行的吗？至少我现在在 DataParallel 的代码中没有看到它们并行的迹象.


## DistributedDataParallel

在切入细节之前，首先需要理清楚 DDP 代码的使用逻辑，不然直接看代码会觉得很困惑．

DDP 与 DP 最大的不同之处在于，DDP 是需要在多个机器（或者多个进程，每个单独的进程可以视作一个虚拟机器）上同时运行的．因此这同一份代码会在多台机器上跑多遍，而 DP 的代码只会在一台机器上跑一遍．

在 DDP 中，所有的计算节点分为 `intra` 和 `inter`.

- `intra` 节点指不同机器或者不同进程（就算是同一个python脚本使用multiprocessing，其各个进程也是独立的）.`intra`节点之间的通讯需要依靠 Pytorch 的 distributed communication package `torch.distributed`，其后端可以是 `tcp, mpi, gloo, nccl`等，但其中`tcp, mpi`支持节点之间的单点通讯，但不支持GPU，而`gloo, nccl`支持GPU, 但只能做一些协同操作，如`broadcast, all_reduce`等，不支持单点通讯．

- `inter`节点是指同一进程管理的多块GPU, 也就是代码中的`self._module_copies`, 这实际上就是 DataParallel 的情况．

但就算在同一台机器上跑多卡的程序，也建议用多进程的方式，也就是把每块卡都看成独立的，这样才能更好的利用 cpu 的计算资源．

不同的后端对应有不同的实现，但大体思路是一致的，这里我们将一个单独的进程视为一个节点：

- 节点内部采用 DataParallel 的方式，计算梯度
- 节点内部梯度计算完毕后，进行节点间梯度 reduce
- 各节点使用 reduce 完毕后的梯度，更新自己的参数，并 copy 给各内部节点

其实，具体实现的细节也有一些值得注意的地方:

- 在节点间传输梯度的过程中，首先将所有的梯度分成多个 buckets (对于`gloo`, 每个bucket的大小大概在10M左右)，然后一个 bucket 中的梯度合并起来一起传输，这样做可以提升传输效率．
- 各节点的梯度 reduce 都放到了 paramters 的 hook (在计算出梯度时会调用的函数)里，但这部分代码涉及到 Pytorch 内部的 backward 机制，所以不是很好看懂．
