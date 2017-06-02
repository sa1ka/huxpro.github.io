---
layout: post
title: Min Stack
---
      
这是LeetCode上的一道Easy难度的题目，问题描述如下：

> Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

>  * push(x) -- Push element x onto stack.
>  * pop() -- Removes the element on top of the stack.
>  * top() -- Get the top element.
>  * getMin() -- Retrieve the minimum element in the stack.

有难度的地方就只在于getMin（）如何在常数时间内实现，一开始我也想不出，之后经过高人指点找到了解决方案

## 解决思路

要想得到更快的时间，总的付出一些代价，常见的就是利用问题的一些特征来优化算法，还有最常用的就是用空间换时间了，首先来看问题，假设我我们已经弄出了一种数据结构能在常数时间内解决这个问题，那么这个数据结构应该满足一些什么样的特性呢？

* 我们对栈的操作都是局限于顶部的，这个数据结构的操作也只需满足局部性即可
* 假设现在的栈在某一个状态，其中的最小值是min，经过一系列的push，pop操作之后又回到了原状态，min值是不会变化的，那我们把每个不同的栈的min值看作它的一个状态，这个状态是随着栈的变化而不断变化的，而且这个2个相同的栈状态也必定是相同的，相当于这个min状态和栈共进退。

总结一下就是个状态我们也可以用一个数据结构来存储，而且这个状态要和栈共进退，很容易想到的就是再用一个栈来存这个min值了，也就是栈中栈！

## 具体实现
平凡的实现方法在这里就不讲了，在这里只讲一种优化的方法，首先还是贴出代码：

### 我是代码

{% highlight cpp %}
void push(int x) {
	if(_size == _capability)
	{
		_capability += INCREASE;
		int *new_stack = new int[_capability];
		memcpy(new_stack, _stack, sizeof(int) * _size);
		delete[] _stack;
		_stack = new_stack;
	}
	_stack[_size++] = x;

	//对栈中栈的操作
	if (_min_stack.size() == 0)
		_min_stack.push(x);
	else if (x <= _min_stack.top())
		_min_stack.push(x);
}

void pop() {
	if (_size == 0)
		return;
	else
		_size--;

	//对栈中栈的操作
	if (_stack[_size] == _min_stack.top())
		_min_stack.pop();
}
{% endhighlight %}


基本想法就是push进来一个数，如果这个数比我现在的min值大，就不用去修改_min_stack,同样的，如果pop出去的东西是比我当前的min值大的，也不用修改_min_stack,其中有个小细节要注意，push进来的值如果和min值相等，也要推入栈中，不然就会出bug，这种方法是对平凡的实现方法上的一个常数级别的优化，还是蛮有趣的。
