<script type="text/javascript" src="https://cdn.bootcss.com/mathjax/2.7.3/latest.js?config=TeX-AMS-MML_HTMLorMML"></script>  
<script type="text/x-mathjax-config">  
    MathJax.Hub.Config({
        tex2jax: {
            inlineMath: [['$$','$$'], ['\\\\(','\\\\)']],
            processEscapes: true
        }
    });
</script>

## This is the reading note of spatial-based graph convolutional neural network

## 来自对《[A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596?context=cs)》的spectral-based GCN部分的翻译

Spectral-based方法在图信号(graph signal processing)处理中已经有了一个非常好的基础。

在Spectral-based的模型中，图通常被假定为无向图。对无向图比较鲁棒的表示方法是归一化的拉普拉斯矩阵（Normalized graph Laplacian matrix）,归一化的拉普拉斯矩阵被定义为：

$i + j$



## Relate Papers

- [M. Defferrard, X. Bresson, and P. Vandergheynst, “Convolutional neural networks on graphs with fast localized spectral filtering,” in Advances in Neural Information Processing Systems, 2016, pp. 3844–3852.](https://arxiv.org/abs/1606.09375)
- [T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in Proceedings of the International Conference on Learning Representations, 2017.](https://arxiv.org/abs/1609.02907)
- [M. Henaff, J. Bruna, and Y. LeCun, “Deep convolutional networks on graph-structured data,” arXiv preprint arXiv:1506.05163, 2015.](https://arxiv.org/abs/1506.05163)
- [R. Li, S. Wang, F. Zhu, and J. Huang, “Adaptive graph convolutional neural networks,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2018, pp. 3546–3553.](https://arxiv.org/abs/1801.03226)
- [R. Levie, F. Monti, X. Bresson, and M. M. Bronstein, “Cayleynets: Graph convolutional neural networks with complex rational spectral filters,” IEEE Transactions on Signal Processing, vol. 67, no. 1, pp. 97–109, 2017.](https://arxiv.org/abs/1705.07664)


