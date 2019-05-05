# Semi-Supervised Clssification with Graph Convolutional Network

## 概述
  
这篇是第一个笔记，作为实验。  
    除去综述那一篇，我的主要工作是围绕着github上面[pytorch geometric](https://github.com/rusty1s/pytorch_geometric)这个项目。这样的主要原因是，在看过综述和几篇模型之后还是觉得对于图网络的模型和原理是认识了，但是对于如何在代码中具体实现，以及具体的图的数据结构是什么样的都不是很清楚。甚至于都不是特别清楚图网络到底能什么？（这个问题在之后的文章里我觉得还是要多聊聊的，有关图网络要解决什么问题，在一些文章里还是有一些分析的）碰巧是这个时候刷到了这个项目，然后就跟着这个项目里涉及的17、18年的20几篇GNN相关的文章来同步理解。这篇_Semi-Supervised Classification with Graph Convolutional Network_是其中的第一篇，主要讲的是GCN，是我看的第一篇明确在讲自己是做spectral domain的图**卷积**的文章。所以从这里开始，最基本的**图卷积**的概念来理解还是比较方便的。    
    在这一篇文章里面，前面简单论述了图卷积自身的基本的原理，并没有太深入的数学部分，直接讲最主要的结论了，这点是比较好的，我们在接下来[模型](##模型)这一部分会仔细推导一下的。值得一提的是，本文中也对于图的结构该如何计算loss进行了简述，这里还是很不错的，我们在[Loss](##Loss)会说明一下。之后要说的点就是实验，实际上对于图网络究竟能解决什么样的问题，我个人觉得还是没有特别大的变化，无论是零几年时候的开篇之作，还是目前对于图网络的这些入门作品，我看到的觉得还是觉得变化不大的，可能是应用这里我读的数量还不够，接下来我觉得还是很有必要讨论一下的。在[实验](##实验)一部分里我们会讨论一下本文的实验，以及我们自己复现的时候做出来的结果和理解。  
    好了，让我们开始吧。  

## 模型

在综述的文章中，我们认识到实际上GNN大都是分成两个部分*propagation step*以及updater。相对来说，updater这部分的意义并不是很大，因为都是一些已有的经验，文章中主要讨论的都是*propagation step*这部分，因为实际上我们如果用类似马尔可夫模型之类的思想去理解图，或者在后面综述篇里面去理解图网络的工作原理的话，如何去用算法模拟出图结构中复杂的关系网络里彼此是如何影响的才是这类算法最美丽的点。那么按照正常的定义的话，GCN的*propagation step*是可以被定义为下式：
$$ H^{(l+1)}=\sigma (\widetilde{D}^{-\frac{1}{2}}\widetilde{A}\widetilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)}) $$  
这里面涉及到一些具体的数学推导，这个公式的由来我们在之后综述部分再去详细推导，但是我们现在可以简单地看一下这个公式，首先这个公式给出了一个迭代的网络传播方式。然后看到$$ \widetilde{D}^{-\frac{1}{2}}\widetilde{A}\widetilde{D}^{-\frac{1}{2}} $$这一部分可以理解为本身图的结构所带来的信息，而$$  D^{(l)} $$则是每一层需要去学习的参数。这里的一个点是，这里就充满了矩阵乘法，而且这里面的两个$$ \widetilde{D} $$矩阵本身是对角线矩阵，那么如何去实现高效的稀疏矩阵乘法以及如何快速地计算出矩阵的幂都可以是我们研究的问题。  
接下来是本文的第一个主要部分，让我们来先讨论一下图上的频域卷积（*spectral convolutions on graphs*）是怎么做的？对于一个单独的向量*x*来说，卷积可以写作
$$ g_{\theta }\star x=Ug_{\theta }U^{T}x  $$
这个公式本身很准确，但是这里面*g*是一个对角矩阵，对角线上每一个元素都是要学习的参数。矩阵*U*是拉氏矩阵的特征向量组成的矩阵，这样的公式显然其计算量是太大了。那么作者考虑的是使用近似，这里借助前几年的研究，用切比雪夫多项式的思想。首先讲前面复杂的矩阵运算部分理解为是对拉氏矩阵特征值的操作，这样的话，再结合[切比雪夫多项式](https://en.wikipedia.org/wiki/Chebyshev_polynomials)，就能将*g*进行近似
$$ g_{\theta^{\prime}}(\Lambda) \approx \sum_{k=0}^{K} \theta_{k}^{\prime} T_{k}(\tilde{\Lambda}) $$
于是这样下来就可以得到卷积的近似为
$$ g_{\theta^{\prime}} \star x \approx \sum_{k=0}^{K} \theta_{k}^{\prime} T_{k}(\tilde{L}) x $$
有理论论证这样的近似之后，整体的计算复杂度就大大降低了。接下来，在本文中，作者采用了K=1，这样就是一个线性的模型。另外，作者说这里近似了特征值的最大值是2，我到现在也是不太清楚为什么，在几篇不同的文章里都已经看过类似的说法，我也不太清楚发生了什么，暂时照着这个想法继续，不行的话我去问几个明白人问问。总之，这样就可以得到前面提到的卷积的有效近似。
$$ g_{\theta^{\prime}} \star x \approx \theta_{0}^{\prime} x+\theta_{1}^{\prime}\left(L-I_{N}\right) x=\theta_{0}^{\prime} x-\theta_{1}^{\prime} D^{-\frac{1}{2}} A D^{-\frac{1}{2}} x $$
这样的话就留下了两组需要学习的参数，为了更加简化，就令两组矩阵互为相反数，这样就得到了下面这个极其简洁的形式。
$$ g_{\theta} \star x \approx \theta\left(I_{N}+D^{-\frac{1}{2}} A D^{-\frac{1}{2}}\right) x $$
为了更加简洁，另外从一个矩阵的角度去思考，就可以得到一个二维的图的频域卷积公式：
$$ Z=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta $$
那么咱们从定性的角度去看这样的网络的话，那么就是三个部分
$$ Z=\sigma (AXW) $$
*A*是图结构的内部信息，*X*是输入，*W*是输入的矩阵，\sigma是激活函数，这样来看就和我们之前熟悉的普通的神经网络很相似了，那么这些道理我们都懂，实际的代码是怎么实现的呢？  
在***pytorch geometric***这个项目里面为了能够提供一个通用的架构，那么他是有一个作为基本信息传输的模型类*message_passing*，其代码如下：
```Python
class MessagePassing(torch.nn.Module):
    '''
    Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
    '''

    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.message_args = inspect.getargspec(self.message)[0][1:]
        self.update_args = inspect.getargspec(self.update)[0][2:]
```
这一部分的具体是想要去实现在pytorch geometric的论文中所说的2017年另一篇文章中提到的信息传输方案。
$$ \vec{x}_{i}^{(k)}=\gamma^{(k)}\left(\vec{x}_{i}^{(k-1)}, \prod_{j \in \mathcal{N}(i)} \phi^{(k)}\left(\vec{x}_{i}^{(k-1)}, \vec{x}_{j}^{(k-1)}, \vec{e}_{i, j}\right)\right) $$  

这个代码里面最主要的其实是*propogation*部分，他给出了一个很有趣的计算方式，我们结合GCN这篇文章来说明一下。
```Python
def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        kwargs['edge_index'] = edge_index
        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.message_args:
            if arg[-2:] in ij.keys():
                tmp = kwargs[arg[:-2]]
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if size[1 - idx] is None:
                            size[1 - idx] = tmp[1 - idx].size(0)
                        if size[1 - idx] != tmp[1 - idx].size(0):
                            raise ValueError(_size_error_msg)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(_size_error_msg)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs[arg])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['size'] = size
        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        out = self.update(out, *update_args)

        return out
```
先不急，我们先来看一下GCN这部分实现的代码。
``` Python
class GCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index = add_self_loops(edge_index, num_nodes)
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
                                            self.improved, x.dtype)
            self.cached_result = edge_index, norm
        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
```
首先回忆一下GCN的卷积公式。
$$ Z=\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} X \Theta $$
在代码里面的*norm*部分，计算的是公式里D^{-1/2}这一块，然后在*forward*的部分里，首先计算*XW*,之后计算出*norm*，再借用*message passing*里面的*propogation*函数，这个函数里面
```Python
    out = self.message(*message_args)
```
```Python
def message(self, x_j, norm):
    return norm.view(-1, 1) * x_j
```
那么这一步就相当于在计算
$$ \tilde{D}^{-\frac{1}{2}} X \Theta $$
紧接下来是*message passing*里面一些有趣的操作，包括作者自己巧妙地利用*scatter*函数的功能去实现了剩下的计算，我们暂时就不在这里赘述了，总之这样的代码的确是实现了GCN的传播公式。而且也是保证了较小的计算量的。  

## Loss

## 实验

## 总结
