---
title: 序列区分性训练 # 标题
date: 2021/10/09 hh:mm:ss # 时间
categories: # 分类
- 人工智能
tags: # 标签
- 语音识别
mathjax: true
---

<h1 style="text-align: center"> 序列区分性训练 </h1>
<div style="text-align: center"><small>陈代君 - 2021</small></div>




## 1. 生成式模型的缺陷

​    语音识别的本质可以用下面的公式进行描述，
$$
\max_{S}\mathbb{P}(S|O)
$$
其中 $$O$$ 表示观测到的语音信号，$$S$$ 表示词序列。传统的语音识别进一步分解上面的公式，
$$
\max_{S} \mathbb{P}(S|O) = \max_{S}\frac{\mathbb{P}_{\theta}(O|S)\mathbb{P}(S)}{\mathbb{P}(O)}
$$
其中，分母 $$\mathbb{P}(O)$$ 在声音信号给定之后是不需要考虑的常量；分子上 $$\mathbb{P}(S)$$ 是语言模型， $$\mathbb{P}_{\theta}(O|S)$$ 是声学模型。 声学模型通常是基于给定的 $$N$$ 个训练数据 $$\{( S_{i}, O_{i})\}_{i=1}^{N}$$ 求得 $$\theta$$ 的极大似然估计。 

​    为了能够理解极大似然估计的缺陷，先回顾一下极大似然估计。 统计的频率学派将参数 $$\theta$$ 视为未知的常量，在模型类型给定的基础上，基于训练数据寻找使得似然最大的 $$\theta$$，记为 $$\hat{\theta}_{MLE}$$， 这个值就是 $$\theta$$ 的极大似然估计。求极大似然估计的过程中，首先需要给定模型类型，例如在语音识别中，通常假定模型为 GMM-HMM 或者 DNN-HMM。因为真实的语音信号并不是由 HMM 生成的，所以在 GMM-HMM 模型的基础上根据似然最大准则找到的最优参数并不一定使得 WER 或者 SER 最小。 换句话说，似然最大准则并不能保证语音识别的错误率最小是极大似然估计的主要缺陷。

​    实际上，$$\mathbb{P}_{\theta}(O|S)$$ 是描述数据分布的生成式模型 (generative model)，即给定词序列，假设观测到的语音信号服从 GMM-HMM 或者 DNN-HMM 模型描述的分布。不同于生成式模型，区分性模型 (discriminative model，也称为判别式模型) 直接对给定观测数据条件下真实类别的分布进行建模。即对 $$\mathbb{P}_{\theta}(S|O)$$ 进行建模。



## 2. MMI 估计 

### 2.1 MMI 估计的基本原理

因为语音信号随机变量 $$\mathbf{O}$$ 和词序列随机变量 $$\mathbf{S}$$ 的互信息如下，
$$
\begin{align}\mathbb{I}(\mathbf{O}, \mathbf{S}) &= \mathbb{E}\log\frac{\mathbb{P}(\mathbf{O},\mathbf{S})}{\mathbb{P}(\mathbf{O})\mathbb{P}(\mathbf{S})}\\
&=\mathbb{H}(\mathbf{S})-\mathbb{H}(\mathbf{S}|\mathbf{O})\end{align}
$$
其中， $$\mathbb{H}(\mathbf{S})$$ 表示词序列随机变量的熵， $$\mathbb{H}(\mathbf{S}|\mathbf{O})$$ 表示条件为观测随机变量的条件熵。

​    同时 $$\mathbb{H}(\mathbf{S})$$ 与待估计模型的参数无关。 所以极大化互信息 $$\mathbb{I}(\mathbf{O}, \mathbf{S})$$ 等价于极小化条件熵 $$\mathbb{H}(\mathbf{S}|\mathbf{O})$$。极小化条件熵意味着在给定观测语音信号随机变量 $$\mathbf{O}$$ 后，极小化词序列随机变量 $$\mathbf{S}$$ 的不确定性。 又因为互信息表达式， 
$$
\mathbb{I}(\mathbf{O}，\mathbf{S}) = \mathbb{E}\log\frac{\mathbb{P}(\mathbf{O},\mathbf{S})}{\mathbb{P}(\mathbf{O})} - \mathbb{E}\log\mathbb{P}(\mathbf{S})
$$
其中 $$\mathbb{P}(\mathbf{S})$$ 不含模型待估参数。 因此，极大化互信息等价于极大化上式右边第一项。所以极大互信息 (MMI, maximum mutual information) 直接求使得 $$\sum_{i=1}^{N}\log\mathbb{P}_{\theta}(S_i|O_i)$$ 极大的 $$\hat{\theta}_{MMI}$$，这个估计被称为 MMI 估计。具体的表达式如下，
$$
\begin{align}
\hat{\theta}_{MMI}  &= \arg\max_{\theta} \sum_{i=1}^{N}\log\mathbb{P}_{\theta}(S_{i}|O_{i})\\
&=\arg \max_{\theta}\sum_{i=1}^{N}\log\frac{\mathbb{P}_{\theta}(O_{i}|S_{i})\mathbb{P}(S_{i})}{\mathbb{P}_{\theta}(O_{i})}\\ &=\arg \max_{\theta}\sum_{i=1}^{N}\log\frac{\mathbb{P}_{\theta}(O_{i}|S_{i})\mathbb{P}(S_{i})}{\sum_{j}\mathbb{P}_{\theta}(O_{i}|S_{i}^{j})\mathbb{P}(S_{i}^{j})}
\end{align}
$$
其中，$$S_{i}^{j}$$ 表示给定观测语音信号 $$O_{i}$$ 条件下可能的词序列。例如，当正确的语音为 "动手学语音识别"，则 “冻手雪余音识别” 可以作为其中一个可能的词序列。

​    为了极大化 MMI 的目标函数，可以增加分子或者减小分母 （MMI 的目标函数可以转化为似然函数乘积的形式）。 增加分子意味着提高给定正确词序列条件下观测语音信号的概率，即声学模型的概率。减小分母等同于减少错误词序列条件下观测语音的概率。综合起来看，增加正确词序列的概率减少错误词序列的概率使得最终的 MMI 估计具有更好的区分性，从而更好的识别出正确的词序列。

### 2.2 MMI 训练算法推导

#### 2.2.1 GMM-HMM

​    理解了 MMI 的基本原理之后，还需要能够快速寻找 MMI 目标函数的优化算法。 相对于极大似然目标函数的优化来说， MMI 的目标函数的优化是更难的。总的来说，有两类算法被用来优化 MMI 的目标函数。第一类是基于梯度的优化算法，例如 GPD (Generalized Probabilistic Descent)。 第二类是更接近 EM 算法的 EB (Extended Baum-Welch) 算法。相对于基于梯度的优化算法，EB 算法具有下面的优势。

- EB 算法利用辅助函数从理论上确保了算法的收敛和高效。
- EB 算法可以处理 HMM 中每个状态到其他状态的转移概率求和等于 1 以及 GMM 各分量权重求和等于 1 这两个等式约束条件。 

​    下面是 GMM-HMM 模型的 EB 算法第 $$k+1$$ 步的迭代公式， 证明的细节参考附录。
$$
\begin{align}
\mu_{jm}^{(k+1)} &= \frac{\theta^{num}_{jm}(\mathbf{O})-\theta_{jm}^{den}(\mathbf{O})+D_{jm}\cdot\mu_{jm}^{(k)}}{\gamma_{jm}^{num}-\gamma_{jm}^{den}+D_{jm}} \\
(\sigma^{2})^{(k+1)}_{jm} &= \frac{\theta_{jm}^{num}(\mathbf{O}^{2})-\theta_{jm}^{den}(\mathbf{O}^2)+D_{jm}\cdot\big((\sigma^{2})_{jm}^{(k)}+(\mu_{jm}^{(k)})^2\big)}{\gamma_{jm}^{num}-\gamma_{jm}^{den}+D_{jm}}-(\mu_{jm}^{(k+1)})^{2}
\end{align}
$$
其中，
$$
\begin{align}
f^{num}(\theta) &= \sum_{i=1}^{N}\log\mathbb{P}_{\theta}(O_{i}|S_{i})\mathbb{P}(S_{i}) = \sum_{i=1}^{N}\log\sum_{a_i}f^{num}_{a_i}(\theta) \\
f^{den}(\theta) &= \sum_{i=1}^{N}\log\sum_{j}\mathbb{P}_{\theta}(O_i|S_i^j)\mathbb{P}_{\theta}(S_i^j) = \sum_{i=1}^{N}\log\sum_{j}\sum_{a_{ij}}f^{den}_{a_{ij}}(\theta)
\end{align}
$$

$$
\begin{align}
\gamma_{ijm}^{num}(t) &= \sum_{a_{i}(t)=j}\frac{f^{num}_{a_{i}}(\theta^{(k)})}{\sum_{y}f^{num}_{y}(\theta^{(k)})},
\gamma_{ijm}^{den}(t) = \sum_{l}\sum_{a_{il}(t)=j}\frac{f_{a_{il}}^{den}(\theta^{(k)})}{\sum_{y}f_{y}^{den}(\theta^{(k)})} \\
\gamma_{jm}^{num} &= \sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\gamma_{ijm}^{num}(t),
\gamma_{jm}^{den} = \sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\gamma_{ijm}^{den}(t) \\ 
\theta_{jm}^{num}(\mathbf{O}) &= \sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\gamma_{ijm}^{num}(t)O_i(t),
\theta_{jm}^{den}(\mathbf{O}) = \sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\gamma_{ijm}^{den}(t)O_i(t)\\
\theta_{jm}^{num}(\mathbf{O}^2) &= \sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\gamma_{ijm}^{num}(t)O_i^2(t), 
\theta_{jm}^{den}(\mathbf{O}^2) = \sum_{i=1}^{N}\sum_{t=1}^{T_{i}}\gamma_{ijm}^{den}(t)O^2_{i}(t)
\end{align}
$$

上面的参数更新公式中，$$\gamma_{ijm}^{den}(t)$$ 的计算需要遍历所有可能的词序列。因为所有可能的词序列数量巨大，所以直接通过上面的更新公式来求解参数的 MMI 估计几乎是不可能的。 为了克服这个计算上的困难，下面两个小节将介绍两种解决方法。

### 2.3 Lattice-based MMI

​    为了加速 MMI 的计算量，早期的工作主要集中在使用 N-best 列表来近似 MMI 分母中所有可能词序列的集合。但是对于非常长的词序列，LM 计算出来的概率都接近于零，导致 N-best 中存储的备选词序列不能很好地近似分母所需的词序列集合 [Chow, 1990] 。另一种逼近 MMI 分母词序列的方案中，对每一个训练词序列生成一个 Lattice（词格），该 Lattice 将会被应用到 MMI 训练且在迭代过程不会被更新 [Povey, 2003]。

​    Lattice-based MMI 的具体实现步骤如下，

---

**实现步骤**

1. 分子部分 Lattice 的构造：基于极大似然准则训练好 GMM-HMM（或基于 CE 准则训练好 DNN-HMM 模型），将训练好的模型应用到正确标准的词序列和音频信号序列上进行状态级别的对齐并生成正确词序列对应的 Lattice。
2. 分母部分 Lattice 的构造: 利用 unigram 语言模型构建出 HCLG, 识别所有的训练例子 $$(O_i, S_i)$$ 并保留识别过程中的 Lattice, $$i=1,\dots,N$$。 尽管在 [Povey, 2003] 文章中，主要利用 bigram 语言模型构建 HCLG，但是论文也提到利用 unigram 语言模型构建 HCLG 会得到更好的实验结果。
3. Lattice-based MMI 的迭代训练: 对于一个训练样本 $$(O_i, S_i)$$，
   - 基于 1. 中构造的分子 Lattice, 利用 unigram 语言模型得到正确路径的概率并计算 $$\gamma_{ijm}^{num}$$ 的前向和后向概率。
   - 基于 2. 中构造的分母 Lattice, 构造正确路径对应的竞争路径集合，并利用 unigram 语言模型计算该竞争路径的概率，类似分子部分计算每条竞争路径的概率后求和得到 $$\gamma_{ijm}^{den}$$ 。
   - 迭代更新模型参数至收敛，得到参数的 MMI 估计。

---

​    上面的实现步骤中，利用两个不同的模型分别对分子和分母部分构建分子 Lattice 以及分母 Lattice。其中分母 Lattice 的构造， 利用了由简单的 unigram 语言模型组成的 HCLG 有限状态转移图。在 MMI 的迭代更新过程中，分母和分子部分的计算都利用了 unigram 语言模型计算路径的概率。

​    下面的表格总结了 [Povey, 2003] 中报告的 Lattice-based MMI 训练过程中的各方面影响，

|                        | 结论                                                         | 解释                                                         |
| ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 区分性训练中的语言模型 | unigram 优于 bigram 和 zerogram (无语言模型) 。              | 尽管在某些测试集上，zerogram 模型或者按比例缩小的 unigram 模型优于 unigram, 但是会增加常用词的错误。 |
| 重新生成 Lattice       | 在训练 MMI 的过程中不需要更新 Lattice。                      | 尽管在 MMI 的训练过程中，基于最新模型重新生成 Lattice 或对齐有最多 0.3% 的绝对 WER 提升，但是由于较高的计算复杂度，所以并不推荐在训练过程中更新 Lattice或对齐。 |
| Lattice 的大小         | 存在一个合适的阈值，使得在不大幅减小最终 MMI 估计识别精度的前提下，可以减小 Lattice 的复杂度。 | 若备选路径 $$p^{'}$$ 满足 $$\log \mathbb{P}(p^{'})/\max_{p}\{\log\mathbb{P}(p)\}<阈值$$，则对该路径进行剪枝。实验结果表明，阈值取100 时，词错误率减小量不超过 0.2%。 |

### 2.4 Lattice-free MMI



## 3. 其他区分性准则

### 3.1 scaled MMI

在 MMI 的目标函数中，分子和分母部分的词序列概率通常采用简单的 unigram 语言模型来近似，因为 unigram 模型忽略了词与词之间的相关性，所以导致备选词序列集合中概率高的词序列数量较少。最终导致 MMI 的目标函数不需要怎么调整声学模型参数就可以达到较大值， 但是却没有提高声学模型对于混淆路径的区分度。为了解决混淆路径较少的问题，引入如下的 scaled MMI 目标函数，
$$
\sum_{i=1}^N\log\frac{\mathbb{P}_{\theta}(O_{i}|S_i)^{\alpha}\mathbb{P}(S_i)^{\alpha}}{\sum_{j}\mathbb{P}_{\theta}(O_i|S_i^j)^{\alpha}\mathbb{P}(S_i^j)^{\alpha}}
$$
其中 $$\alpha$$ 被称为尺度因子 (scale factor)，$$0<\alpha<1 $$。

加入尺度因子之后，能够缩小备选词序列集合中词序列之间的概率差异，从而增加混淆路径的数量。使得最大化 scaled MMI 目标函数能够得到更具有区分性的声学模型 $$\mathbb{P}_{\theta}(O_i|S_i)$$。

<p align = "center">
    <img height="400px" src = "./images/scaleFactor.PNG">
    图 3-1 尺度因子对词序列概率的作用

为了直观说明尺度因子$$\alpha$$ 的作用，图 3-1 给出了四个不同的尺度因子条件下词序列概率 $$\mathbb{P}(S_i^j)^{\alpha}$$ 的变化。 假设 $$\mathbb{P}(S_i^1)=0.9, \mathbb{P}(S_i^2)=0.8$$，两条红色的水平虚线和纵轴的交点间距表示 $$\alpha=1$$ 时两个词序列概率的差；而两条蓝色的水平虚线和纵轴的交点间距表示 $$\alpha=0.25$$ 时两个词序列概率的差。从图 3-1 中可以看到，蓝色的交点间距明显小于红色的交点间距，所以应用更小的尺度因子可以减小词序列概率间的差异。

### 3.2 BMMI

增强的 MMI (BMMI, Boosted MMI) 目标函数为,
$$
\sum_{i=1}^{N}\log\frac{\mathbb{P}_{\theta}(O_i|S_i)^{\alpha}\mathbb{P}(S_i)^{\alpha}}{\sum_{j}\mathbb{P}_{\theta}(O_i|S_i^j)^{\alpha}\mathbb{P}(S_i^{j})^{\alpha}\exp\{-b\cdot A(S_i^j,S_i)\}}
$$
其中 $$\alpha$$ 是尺度因子，b 是非负的增强因子 (boosting factor), $$A(S_i^j, S_i)$$ 度量备选文本 $$S_i^j$$ 和正确文本 $S_i$ 之间的匹配程度。常用已经训练好的 GMM-HMM 或者 DNN-HMM 把音频信号 $$O_i$$ 和 $$S_i$$ 以及 $$S_i^j$$ 分别进行帧-音素级别的对齐，然后统计两个帧-音素对齐中结果相同的帧数。$$S_i^j$$ 和 $$S_i$$ 匹配程度越高， 该备选词序列的重要度越低；反之 $$S_i^j$$ 和 $$S_i$$ 匹配程度越低，该备选词序列的重要度越高。最大化 BMMI 目标函数， 匹配程度较低的备选词序列相对于匹配程度较高的备选词序列来说，声学模型 $$\mathbb{P}_{\theta}(O_i|S_i^j)$$ 的取值会变小。这就使得经过序列区分训练后的声学模型更倾向于选择更贴近真实词序列的备选答案。

### 3.3 MPE/MWE/sMBR

BMMI 在 MMI 目标函数分母的每一项增加了权重，如果在 MMI 目标函数分子增加权重，修改目标函数如下，
$$
\sum_{i=1}^{N}\frac{\sum_j\mathbb{P}_{\theta}(O_i|S_i^j)^{\alpha}\mathbb{P}(S_i^j)^{\alpha}\cdot A(S_i^j, S_i)}{\sum_{j}\mathbb{P}_{\theta}(O_i|S_i^j)^{\alpha}\mathbb{P}(S_i^j)^{\alpha}}
$$
需要特别注意上面的目标函数是N个训练样本上概率求和，而不是取了对数之后再求和。如果上面公式中的 $$A(S_i^j,S_i)$$ 和 BMMI 一样，都是度量正确文本 $$S_i$$ 和备选文本 $$S_i^j$$ 之间帧-音素级别的匹配程度，则该目标函数被称为 MPE (Minimum Phone Error) 准则。 

如果上面公式中的 $$A(S_i^j, S_i)$$ 度量备选文本 $$S_i$$ 和正确文本 $$S_i^j$$ 对齐之后正确词 (字) 的个数，则该目标函数被称为 MWE (Minimum Word Error) 准则。文本之间的对齐可以参考编辑距离的计算，或者考虑 WER 的计算过程。

如果上面公式中的 $$A(S_i^j, S_i)$$ 度量备选文本 $$S_i$$ 和正确文本 $$S_i^j$$ 在帧-状态级别的匹配程度，即利用训练好的 GMM-HMM 或者 DNN-HMM 对音频信号 $$O_i$$ 和 $$S_i$$ 以及 $$S_i^j$$ 分别进行帧-状态级别的对齐，然后统计两个帧-状态对齐中结果相同的帧数， 则该目标函数被称为 sMBR (state-Minimum Bayesian Risk)。 

不论是 MPE，MWE 或者 sMBR， 都是给予了和正确文本相似的备选文本较高的权重，优化目标函数之后得到的声学模型 $$\mathbb{P}_{\theta}(O_i|S_i^j)$$ 在相似度高的文本上给予较高的声学得分，而在相似度低的文本上给予较低的声学得分。

对比 MPE 和 MWE 的目标函数，理论上，当训练数据量趋于无穷， MWE 的目标函数应该逼近模型的泛化 WER， 所以最小化 MWE 的目标函数应该和最小化 WER 的目标更匹配。  但是，在目前能够得到的训练集上，实验结果表明 MPE 在测试集上的表现比 MWE 的更好一些 [Povey, 2003]。



## 4. Kaldi 实现



## 5. 本章小结



## 6. 参考资料



## 7. 附录

## Text Formatting

Regular, **bold**, *italic*, ~~strike~~, ==hightlight==, `inline-code`,*emphasis^,<!--comment-->,

## Cites

> [Povey ]e



## Code:

```js
import someCode from 'someLibrary';

```

## Links

[This is a link](www.google.com)

## Footnote

Some thing 

## Superscripts

Example^1^

Example~2~
