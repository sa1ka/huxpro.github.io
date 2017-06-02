---
layout: post
title: C++ STL中 next permutaion 的原理
---
今天在学习C++ STL的时候发现了一个很有趣而且很实用的函数，也就是针对顺序容器的一个排列函数：next_permutation，其函数原型如下：
next_permutation的函数原型如下：
{% highlight c++ %}
template<class BidirectionalIterator>
bool next_permutation(
      BidirectionalIterator _First, 
      BidirectionalIterator _Last
);
template<class BidirectionalIterator, class BinaryPredicate>
bool next_permutation(
      BidirectionalIterator _First, 
      BidirectionalIterator _Last,
      BinaryPredicate _Comp
 );
{% endhighlight %}
函数的作用就是得到比当前排列“大”的下一个排列，关于这个“大”，这里用数字排列比较容易，比如123的下一排列就是132，再下一个排列是213。同样的可以推广到其他任何能进行比较大小的元素的序列。

在STL里面函数实现的原理是这样的：
在当前序列中，从尾端向前寻找两个相邻元素，前一个记为*i，后一个记为*t，并且满足*i < *t。然后再从尾端寻找另一个元素*j，如果满足*i < *j，即将第i个元素与第j个元素对调，并将第t个元素之后（包括t）的所有元素颠倒排序，即求出下一个序列了。

我一开始看到这个原理的时候还没能理解到底是为什么，之后自己想了一会才真正理解了这个算法的含义。下面结合一个具体的例子来解释一下这个算法：
比如有一个排列13542，记为x，首先从尾向前寻找两个相邻的元素，而且前一个小于后一个，在这里找到了35，而且同时保证了5及5之后的元素是保持降序排列的，因为对于5之后的相邻对都有前一个元素大于等于后一个元素，记这个子排列为y，可以知道y排列已经是最大状态了，因此需要对y之前的元素进行改动才能使得x排列进一步增大，并且要使得这个增量最小，因此改动只能选择5之前紧跟的那个3，我们需要从y中选取一个元素来与3进行对换，那么肯定需要选一个尽量小的元素，也就是从后往前找到第一个大于3的元素4，替换之后排列变成了14532，在这里可以看到序列y'(532)依旧是保持降序的，这是不是偶然而是必然的，那么在最后我们还需要进行最后一边操作，也就是把y'反转，得到14235，也就得到了最终的结果。

其实与next_permutaion对应的还有一个函数是prev_permutation，也就是找比当前排列“小”的前一个排列，原理与next_permutation是一样的。
