//! 栈数据结构实现
//!
//! 提供基于向量的高效栈实现，支持 LIFO 操作

use super::Vector;
use std::fmt::{self, Debug};
use std::iter::FromIterator;
use std::ops::RangeBounds;

/// 栈数据结构
///
/// # 类型参数
///
/// * `T` - 存储的元素类型
///
/// # 示例
///
/// ```
/// use dstructs::core::collections::Stack;
///
/// let mut stack = Stack::new();
/// stack.push(1);
/// stack.push(2);
/// assert_eq!(stack.pop(), Some(2));
/// assert_eq!(stack.peek(), Some(&1));
///
/// // 使用迭代器
/// stack.push(2);
/// stack.push(3);
/// let sum: i32 = stack.iter().sum(); // 从栈顶到栈底迭代：3 + 2 + 1 = 6
/// assert_eq!(sum, 6);
/// ```
#[derive(Clone)]
pub struct Stack<T> {
    // 使用 Vector 作为底层存储
    data: Vector<T>,
}

/// 栈的不可变迭代器
///
/// 按照从栈顶到栈底的顺序迭代元素
pub struct Iter<'a, T> {
    // 使用切片迭代器，但是反向迭代以保持栈的 LIFO 顺序
    iter: std::iter::Rev<std::slice::Iter<'a, T>>,
}

/// 栈的可变迭代器
///
/// 按照从栈顶到栈底的顺序迭代元素
pub struct IterMut<'a, T> {
    // 使用切片迭代器，但是反向迭代以保持栈的 LIFO 顺序
    iter: std::iter::Rev<std::slice::IterMut<'a, T>>,
}

/// 用于批量移除元素的迭代器
pub struct Drain<T> {
    // 存储已经移除的元素
    elements: Vec<T>,
}

impl<T> Stack<T> {
    /// 创建一个新的空栈
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Vector::new(),
        }
    }

    /// 创建一个具有指定容量的空栈
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vector::with_capacity(capacity),
        }
    }

    /// 将元素压入栈顶
    #[inline]
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// 弹出栈顶元素
    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    /// 查看栈顶元素但不移除
    #[inline]
    pub fn peek(&self) -> Option<&T> {
        self.data.last()
    }

    /// 查看栈顶元素的可变引用但不移除
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut T> {
        self.data.last_mut()
    }

    /// 返回栈中的元素数量
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 检查栈是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 清空栈
    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// 返回栈的容量
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// 保留指定容量的空间
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// 收缩栈的容量以适应当前元素数量
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.data.shrink_to_fit();
    }

    /// 返回一个从栈顶到栈底的迭代器
    ///
    /// # 示例
    ///
    /// ```
    /// use dstructs::core::collections::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// stack.push(3);
    ///
    /// let mut iter = stack.iter();
    /// assert_eq!(iter.next(), Some(&3)); // 栈顶
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&1)); // 栈底
    /// assert_eq!(iter.next(), None);
    /// ```
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.data.iter().rev(),
        }
    }

    /// 返回一个从栈顶到栈底的可变迭代器
    ///
    /// # 示例
    ///
    /// ```
    /// use dstructs::core::collections::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// stack.push(3);
    ///
    /// // 将所有元素加倍
    /// for x in stack.iter_mut() {
    ///     *x *= 2;
    /// }
    ///
    /// assert_eq!(stack.pop(), Some(6)); // 栈顶
    /// assert_eq!(stack.pop(), Some(4));
    /// assert_eq!(stack.pop(), Some(2)); // 栈底
    /// ```
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            iter: self.data.iter_mut().rev(),
        }
    }

    /// 返回一个包含所有元素的切片
    ///
    /// 注意：切片中的元素顺序是从栈底到栈顶
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// 返回一个包含所有元素的可变切片
    ///
    /// 注意：切片中的元素顺序是从栈底到栈顶
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// 移除指定范围的元素并返回迭代器
    ///
    /// 注意：返回的迭代器按照栈的顺序（从高地址到低地址）产出元素
    ///
    /// # 示例
    ///
    /// ```
    /// use dstructs::core::collections::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.extend(0..5); // 添加 [0, 1, 2, 3, 4]
    ///
    /// // 移除中间的元素
    /// let drained: Vec<_> = stack.drain(1..4).collect();
    /// assert_eq!(drained, vec![1, 2, 3]); // 注意顺序是从栈顶到栈底
    ///
    /// // 检查剩余元素
    /// assert_eq!(stack.pop(), Some(4));
    /// assert_eq!(stack.pop(), Some(0));
    /// ```
    #[inline]
    pub fn drain<R>(&mut self, range: R) -> Drain<T>
    where
        R: RangeBounds<usize>,
    {
        // 收集要移除的元素
        let mut elements: Vec<T> = self.data.drain(range).collect();
        // 反转元素以保持栈的顺序
        elements.reverse();
        Drain { elements }
    }
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<T> Iterator for Drain<T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.elements.pop()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.elements.len();
        (len, Some(len))
    }
}

// 实现 IntoIterator trait，使 Stack 可以直接用于 for 循环
impl<'a, T> IntoIterator for &'a Stack<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Stack<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Debug> Debug for Stack<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stack").field("data", &self.data).finish()
    }
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FromIterator<T> for Stack<T> {
    /// 从迭代器创建栈
    ///
    /// 注意：元素的顺序将与迭代器顺序相反，以保持栈的 LIFO 特性
    ///
    /// # 示例
    ///
    /// ```
    /// use dstructs::core::collections::Stack;
    ///
    /// let vec = vec![1, 2, 3];
    /// let mut stack: Stack<_> = vec.into_iter().collect();
    ///
    /// assert_eq!(stack.pop(), Some(3)); // 最后一个元素在栈顶
    /// assert_eq!(stack.pop(), Some(2));
    /// assert_eq!(stack.pop(), Some(1));
    /// ```
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut stack = Stack::new();
        stack.extend(iter);
        stack
    }
}

impl<T> Extend<T> for Stack<T> {
    /// 扩展栈，添加迭代器中的所有元素
    ///
    /// 元素将按照迭代器的顺序压入栈中
    ///
    /// # 示例
    ///
    /// ```
    /// use dstructs::core::collections::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    ///
    /// stack.extend(vec![2, 3, 4]);
    /// assert_eq!(stack.pop(), Some(4)); // 最后添加的元素在栈顶
    /// assert_eq!(stack.pop(), Some(3));
    /// assert_eq!(stack.pop(), Some(2));
    /// assert_eq!(stack.pop(), Some(1));
    /// ```
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.data.extend(iter);
    }
}

impl<T> From<Vec<T>> for Stack<T> {
    /// 从 Vec 创建 Stack
    ///
    /// 注意：Vec 的最后一个元素将成为栈顶
    ///
    /// # 示例
    ///
    /// ```
    /// use dstructs::core::collections::Stack;
    ///
    /// let vec = vec![1, 2, 3];
    /// let mut stack = Stack::from(vec);
    ///
    /// assert_eq!(stack.pop(), Some(3)); // Vec 的最后一个元素在栈顶
    /// assert_eq!(stack.pop(), Some(2));
    /// assert_eq!(stack.pop(), Some(1));
    /// ```
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self {
            data: Vector::from(vec),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_basic_operations() {
        let mut stack = Stack::new();

        // 测试压入和弹出
        stack.push(1);
        stack.push(2);
        stack.push(3);

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
        assert_eq!(stack.pop(), None);

        // 测试空栈操作
        assert!(stack.is_empty());
        assert_eq!(stack.peek(), None);
    }

    #[test]
    fn test_stack_peek() {
        let mut stack = Stack::new();

        stack.push(1);
        assert_eq!(stack.peek(), Some(&1));
        assert_eq!(stack.peek_mut(), Some(&mut 1));

        // 修改栈顶元素
        if let Some(top) = stack.peek_mut() {
            *top = 2;
        }

        assert_eq!(stack.pop(), Some(2));
    }

    #[test]
    fn test_stack_capacity() {
        let mut stack = Stack::with_capacity(10);

        assert!(stack.capacity() >= 10);

        // 测试扩容
        for i in 0..20 {
            stack.push(i);
        }

        assert!(stack.capacity() >= 20);

        // 测试收缩
        for _ in 0..15 {
            stack.pop();
        }

        stack.shrink_to_fit();
        assert!(stack.capacity() >= 5);
    }

    #[test]
    fn test_iterator() {
        let mut stack = Stack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);

        // 测试不可变迭代器
        let mut iter = stack.iter();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);

        // 测试 for 循环语法
        let mut sum = 0;
        for &x in &stack {
            sum += x;
        }
        assert_eq!(sum, 6);

        // 测试可变迭代器
        for x in &mut stack {
            *x *= 2;
        }

        assert_eq!(stack.pop(), Some(6));
        assert_eq!(stack.pop(), Some(4));
        assert_eq!(stack.pop(), Some(2));
    }

    #[test]
    fn test_as_slice() {
        let mut stack = Stack::new();
        stack.push(1);
        stack.push(2);
        stack.push(3);

        // 测试 as_slice
        let slice = stack.as_slice();
        assert_eq!(slice, &[1, 2, 3]); // 注意：切片顺序是从栈底到栈顶

        // 测试 as_mut_slice
        let mut_slice = stack.as_mut_slice();
        mut_slice[0] *= 2;
        mut_slice[1] *= 2;
        mut_slice[2] *= 2;

        assert_eq!(stack.pop(), Some(6));
        assert_eq!(stack.pop(), Some(4));
        assert_eq!(stack.pop(), Some(2));
    }

    #[test]
    fn test_from_iterator() {
        let vec = vec![1, 2, 3];
        let mut stack: Stack<_> = vec.into_iter().collect();

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
    }

    #[test]
    fn test_extend() {
        let mut stack = Stack::new();
        stack.push(1);

        // 测试扩展
        stack.extend(vec![2, 3, 4]);
        assert_eq!(stack.len(), 4);

        // 验证顺序
        assert_eq!(stack.pop(), Some(4));
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));

        // 测试空迭代器扩展
        stack.extend(std::iter::empty::<i32>());
        assert!(stack.is_empty());

        // 测试多次扩展
        stack.extend(0..3);
        stack.extend(3..5);
        assert_eq!(stack.len(), 5);
        for i in (0..5).rev() {
            assert_eq!(stack.pop(), Some(i));
        }
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![1, 2, 3];
        let mut stack = Stack::from(vec);

        assert_eq!(stack.len(), 3);
        assert_eq!(stack.pop(), Some(3));
        assert_eq!(stack.pop(), Some(2));
        assert_eq!(stack.pop(), Some(1));
    }

    #[test]
    fn test_drain() {
        let mut stack = Stack::new();
        stack.extend(0..5); // [0, 1, 2, 3, 4]

        // 测试中间范围
        let drained: Vec<_> = stack.drain(1..4).collect();
        assert_eq!(drained, vec![1, 2, 3]); // 从栈顶到栈底的顺序
        assert_eq!(stack.len(), 2);
        assert_eq!(stack.pop(), Some(4));
        assert_eq!(stack.pop(), Some(0));

        // 测试空范围
        stack.extend(0..3); // [0, 1, 2]
        let drained: Vec<_> = stack.drain(1..1).collect();
        assert!(drained.is_empty());
        assert_eq!(stack.len(), 3);

        // 测试全范围
        let drained: Vec<_> = stack.drain(..).collect();
        assert_eq!(drained, vec![0, 1, 2]);
        assert!(stack.is_empty());
    }
}
