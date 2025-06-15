//! 优化的向量实现
//!
//! 提供一个针对嵌入式环境优化的向量实现

use smallvec::SmallVec;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut, Index, IndexMut, Range, RangeBounds};

/// 针对小数组优化的向量类型
#[derive(Debug, Clone)]
#[repr(C, align(16))] // SIMD 对齐
pub struct Vector<T> {
    inner: SmallVec<[T; 16]>,
}

/// 用于批量移除元素的迭代器
pub struct Drain<'a, T: 'a> {
    vec: &'a mut Vector<T>,
    range: Range<usize>,
}

impl<T> Vector<T> {
    /// 创建新的向量
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            inner: SmallVec::new(),
        }
    }

    /// 创建指定容量的向量
    #[inline(always)]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: SmallVec::with_capacity(capacity),
        }
    }

    /// 添加元素到向量末尾
    #[inline(always)]
    pub fn push(&mut self, value: T) {
        // 预分配增长策略优化
        if self.len() == self.capacity() {
            let new_cap = std::cmp::max(self.capacity() * 2, 4);
            self.reserve(new_cap - self.capacity());
        }
        self.inner.push(value);
    }

    /// 从向量末尾移除并返回元素
    #[inline(always)]
    pub fn pop(&mut self) -> Option<T> {
        self.inner.pop()
    }

    /// 返回向量的容量
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// 清空向量
    #[inline(always)]
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// 检查向量是否为空
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 返回向量中的元素数量
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// 预留至少能容纳 `additional` 个元素的空间
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        let new_cap = self.len() + additional;
        if new_cap > self.capacity() {
            // 使用指数增长策略
            let aligned_cap = (new_cap + 15) & !15; // 16字节对齐
            self.inner.reserve(aligned_cap - self.len());
        }
    }

    /// 将向量缩小到最小所需容量
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        if self.spilled() {
            self.inner.shrink_to_fit();
        }
    }

    /// 获取原始指针
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// 获取可变原始指针
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    /// 批量插入元素
    #[inline]
    pub fn extend_from_slice(&mut self, other: &[T])
    where
        T: Copy,
    {
        self.inner.extend_from_slice(other);
    }

    /// 追加另一个切片中的所有元素（克隆版本）
    #[inline]
    pub fn extend_from_slice_cloned(&mut self, other: &[T])
    where
        T: Clone,
    {
        self.inner.extend(other.iter().cloned());
    }

    /// 批量移除元素
    #[inline]
    pub fn drain<R>(&mut self, range: R) -> impl Iterator<Item = T> + '_
    where
        R: RangeBounds<usize>,
    {
        self.inner.drain(range)
    }

    /// 检查是否使用堆内存
    #[inline]
    pub fn spilled(&self) -> bool {
        self.inner.spilled()
    }

    /// 获取指定范围的切片
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        self.inner.get(index)
    }

    /// 获取指定范围的可变切片
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.inner.get_mut(index)
    }

    /// 获取指定范围的切片
    #[inline]
    pub fn slice(&self, range: Range<usize>) -> Option<&[T]> {
        if range.start <= range.end && range.end <= self.len() {
            Some(&self.inner[range])
        } else {
            None
        }
    }

    /// 获取指定范围的可变切片
    #[inline]
    pub fn slice_mut(&mut self, range: Range<usize>) -> Option<&mut [T]> {
        let len = self.len();
        if range.start <= range.end && range.end <= len {
            Some(&mut self.inner[range])
        } else {
            None
        }
    }

    /// 在指定位置插入元素
    #[inline]
    pub fn insert(&mut self, index: usize, value: T) {
        self.inner.insert(index, value);
    }

    /// 移除指定位置的元素
    #[inline]
    pub fn remove(&mut self, index: usize) -> T {
        self.inner.remove(index)
    }

    /// 保留满足条件的元素
    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.inner.retain(|x| f(x));
    }

    /// 将向量截断到指定长度
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }

    /// 将所有元素移动到另一个向量中
    #[inline]
    pub fn append(&mut self, other: &mut Self) {
        let other_vec = std::mem::take(&mut other.inner);
        self.inner.extend(other_vec);
    }

    /// 分割向量，返回指定长度的前缀
    #[inline]
    pub fn split_off(&mut self, at: usize) -> Self {
        let mut other = Vector::new();
        if at < self.len() {
            other.extend(self.inner.drain(at..));
        }
        other
    }
}

impl<T> Default for Vector<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Deref for Vector<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.inner.as_slice()
    }
}

impl<T> DerefMut for Vector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut_slice()
    }
}

impl<T> From<Vec<T>> for Vector<T> {
    fn from(vec: Vec<T>) -> Self {
        Self {
            inner: SmallVec::from_vec(vec),
        }
    }
}

impl<T> Extend<T> for Vector<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.inner.extend(iter);
    }
}

impl<T> FromIterator<T> for Vector<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut vector = Self::with_capacity(lower);
        vector.extend(iter);
        vector
    }
}

impl<T> IntoIterator for Vector<T> {
    type Item = T;
    type IntoIter = smallvec::IntoIter<[T; 16]>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Vector<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vector<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index]
    }
}

impl<T> AsRef<[T]> for Vector<T> {
    fn as_ref(&self) -> &[T] {
        self.inner.as_ref()
    }
}

impl<T> AsMut<[T]> for Vector<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.inner.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let vec: Vector<i32> = Vector::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
        assert!(!vec.spilled());

        let vec = Vector::<i32>::with_capacity(10);
        assert_eq!(vec.len(), 0);
        assert!(vec.capacity() >= 10);
    }

    #[test]
    fn test_vector_push() {
        let mut vec: Vector<i32> = Vector::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);
    }

    #[test]
    fn test_vector_pop() {
        let mut vec = Vector::new();
        vec.push(1);
        vec.push(2);

        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn test_vector_clear() {
        let mut vec = Vector::new();
        vec.push(1);
        vec.push(2);
        vec.clear();

        assert!(vec.is_empty());
        assert_eq!(vec.len(), 0);
    }

    #[test]
    fn test_vector_from_vec() {
        let std_vec = vec![1, 2, 3];
        let vec = Vector::from(std_vec);

        assert_eq!(vec.len(), 3);
        assert_eq!(&*vec, &[1, 2, 3]);
    }

    #[test]
    fn test_vector_extend() {
        let mut vec = Vector::new();
        vec.extend(0..3);

        assert_eq!(vec.len(), 3);
        assert_eq!(&*vec, &[0, 1, 2]);
    }

    #[test]
    fn test_vector_spill() {
        let mut vec: Vector<i32> = Vector::new();
        assert!(!vec.spilled());

        // 添加超过内联存储容量的元素
        for i in 0..32 {
            vec.push(i);
        }

        // 验证是否溢出到堆上
        assert!(vec.spilled());
    }

    #[test]
    fn test_from_iterator() {
        let vec: Vector<i32> = (0..5).collect();
        assert_eq!(&*vec, &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_into_iterator() {
        let vec: Vector<i32> = (0..5).collect();
        let sum: i32 = vec.into_iter().sum();
        assert_eq!(sum, 10);
    }

    #[test]
    fn test_slice_operations() {
        let mut vec = Vector::new();
        vec.extend(0..5);

        assert_eq!(vec.slice(1..4), Some(&[1, 2, 3][..]));
        assert_eq!(vec.get(5), None);

        if let Some(value) = vec.get_mut(1) {
            *value = 10;
        }
        assert_eq!(&*vec, &[0, 10, 2, 3, 4]);
    }

    #[test]
    fn test_insert_remove() {
        let mut vec = Vector::new();
        vec.extend(0..3);

        vec.insert(1, 10);
        assert_eq!(&*vec, &[0, 10, 1, 2]);

        assert_eq!(vec.remove(1), 10);
        assert_eq!(&*vec, &[0, 1, 2]);
    }

    #[test]
    fn test_retain() {
        let mut vec = Vector::new();
        vec.extend(0..5);

        vec.retain(|&x| x % 2 == 0);
        assert_eq!(&*vec, &[0, 2, 4]);
    }

    #[test]
    fn test_append() {
        let mut vec1 = Vector::new();
        let mut vec2 = Vector::new();

        vec1.extend(0..3);
        vec2.extend(3..6);

        vec1.append(&mut vec2);
        assert_eq!(&*vec1, &[0, 1, 2, 3, 4, 5]);
        assert!(vec2.is_empty());
    }

    #[test]
    fn test_split_off() {
        let mut vec = Vector::new();
        vec.extend(0..5);

        let vec2 = vec.split_off(2);
        assert_eq!(&*vec, &[0, 1]);
        assert_eq!(&*vec2, &[2, 3, 4]);
    }

    #[test]
    fn test_performance_critical_path() {
        let mut vec = Vector::with_capacity(1000);

        // 测试连续 push 的性能
        for i in 0..1000 {
            vec.push(i);
        }
        assert_eq!(vec.len(), 1000);

        // 测试批量操作性能
        let data: Vec<i32> = (0..1000).collect();
        let mut vec = Vector::new();
        vec.extend_from_slice(&data);
        assert_eq!(vec.len(), 1000);

        // 测试内存对齐
        assert_eq!(std::mem::align_of_val(&vec) % 16, 0);
    }
}
