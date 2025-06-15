#![allow(unsafe_code)]
//! 双向链表实现
//!
//! 提供高效的双向链表数据结构，支持O(1)首尾操作和迭代器

use std::marker::PhantomData;
use std::ptr::NonNull;

/// 链表节点结构
struct Node<T> {
    /// 节点数据
    data: T,
    /// 前驱节点
    prev: Option<NonNull<Node<T>>>,
    /// 后继节点
    next: Option<NonNull<Node<T>>>,
}

/// 双向链表
pub struct List<T> {
    /// 头节点
    head: Option<NonNull<Node<T>>>,
    /// 尾节点
    tail: Option<NonNull<Node<T>>>,
    /// 链表长度
    len: usize,
    /// 标记类型参数的生命周期
    marker: PhantomData<T>,
}

impl<T> Node<T> {
    /// 创建新节点
    fn new(data: T) -> Self {
        Node {
            data,
            prev: None,
            next: None,
        }
    }

    /// 分配新节点并返回裸指针
    fn into_ptr(self) -> NonNull<Self> {
        let node = Box::new(self);
        NonNull::new(Box::into_raw(node)).unwrap()
    }
}

impl<T> List<T> {
    /// 创建空链表
    pub fn new() -> Self {
        List {
            head: None,
            tail: None,
            len: 0,
            marker: PhantomData,
        }
    }

    /// 返回链表长度
    pub fn len(&self) -> usize {
        self.len
    }

    /// 判断链表是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 在链表头部插入元素
    pub fn push_front(&mut self, data: T) {
        let new_node = Node::new(data).into_ptr();

        // 安全性：new_node 是有效的非空指针
        unsafe {
            (*new_node.as_ptr()).next = self.head;
            if let Some(head) = self.head {
                (*head.as_ptr()).prev = Some(new_node);
            } else {
                // 如果链表为空，新节点同时是尾节点
                self.tail = Some(new_node);
            }
            self.head = Some(new_node);
        }
        self.len += 1;
    }

    /// 在链表尾部插入元素
    pub fn push_back(&mut self, data: T) {
        let new_node = Node::new(data).into_ptr();

        // 安全性：new_node 是有效的非空指针
        unsafe {
            (*new_node.as_ptr()).prev = self.tail;
            if let Some(tail) = self.tail {
                (*tail.as_ptr()).next = Some(new_node);
            } else {
                // 如果链表为空，新节点同时是头节点
                self.head = Some(new_node);
            }
            self.tail = Some(new_node);
        }
        self.len += 1;
    }

    /// 从链表头部移除元素
    pub fn pop_front(&mut self) -> Option<T> {
        self.head.map(|head| unsafe {
            let node = Box::from_raw(head.as_ptr());
            self.head = node.next;

            if let Some(new_head) = self.head {
                (*new_head.as_ptr()).prev = None;
            } else {
                self.tail = None;
            }

            self.len -= 1;
            node.data
        })
    }

    /// 从链表尾部移除元素
    pub fn pop_back(&mut self) -> Option<T> {
        self.tail.map(|tail| unsafe {
            let node = Box::from_raw(tail.as_ptr());
            self.tail = node.prev;

            if let Some(new_tail) = self.tail {
                (*new_tail.as_ptr()).next = None;
            } else {
                self.head = None;
            }

            self.len -= 1;
            node.data
        })
    }

    /// 获取前向迭代器
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            head: self.head,
            tail: self.tail,
            len: self.len,
            marker: PhantomData,
        }
    }

    /// 获取可变前向迭代器
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        IterMut {
            head: self.head,
            tail: self.tail,
            len: self.len,
            marker: PhantomData,
        }
    }
}

/// 前向迭代器
pub struct Iter<'a, T> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
    marker: PhantomData<&'a Node<T>>,
}

/// 可变前向迭代器
pub struct IterMut<'a, T> {
    head: Option<NonNull<Node<T>>>,
    tail: Option<NonNull<Node<T>>>,
    len: usize,
    marker: PhantomData<&'a mut Node<T>>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.head.map(|node| unsafe {
                let node = &*node.as_ptr();
                self.head = node.next;
                self.len -= 1;
                &node.data
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.head.map(|node| unsafe {
                let node = &mut *node.as_ptr();
                self.head = node.next;
                self.len -= 1;
                &mut node.data
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.tail.map(|node| unsafe {
                let node = &*node.as_ptr();
                self.tail = node.prev;
                self.len -= 1;
                &node.data
            })
        }
    }
}

impl<'a, T> DoubleEndedIterator for IterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len == 0 {
            None
        } else {
            self.tail.map(|node| unsafe {
                let node = &mut *node.as_ptr();
                self.tail = node.prev;
                self.len -= 1;
                &mut node.data
            })
        }
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}
impl<'a, T> ExactSizeIterator for IterMut<'a, T> {}

// 为 List 实现 IntoIterator trait
impl<T> IntoIterator for List<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self)
    }
}

impl<'a, T> IntoIterator for &'a List<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut List<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// 所有权迭代器
pub struct IntoIter<T>(List<T>);

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0.len, Some(self.0.len))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.pop_back()
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

// 实现 Drop trait 以释放所有节点
impl<T> Drop for List<T> {
    fn drop(&mut self) {
        while self.pop_front().is_some() {}
    }
}

// 实现 Default trait
impl<T> Default for List<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop_front() {
        let mut list = List::new();
        assert!(list.is_empty());

        list.push_front(1);
        list.push_front(2);
        assert_eq!(list.len(), 2);

        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn test_push_pop_back() {
        let mut list = List::new();

        list.push_back(1);
        list.push_back(2);
        assert_eq!(list.len(), 2);

        assert_eq!(list.pop_back(), Some(2));
        assert_eq!(list.pop_back(), Some(1));
        assert_eq!(list.pop_back(), None);
    }

    #[test]
    fn test_mixed_operations() {
        let mut list = List::new();

        list.push_front(2);
        list.push_back(3);
        list.push_front(1);

        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.pop_back(), Some(3));
        assert_eq!(list.pop_front(), Some(2));
        assert!(list.is_empty());
    }

    #[test]
    fn test_iterator() {
        let mut list = List::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);

        let mut iter = list.iter().rev();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_iterator_mut() {
        let mut list = List::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        for x in &mut list {
            *x += 10;
        }

        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(&11));
        assert_eq!(iter.next(), Some(&12));
        assert_eq!(iter.next(), Some(&13));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_into_iterator() {
        let mut list = List::new();
        list.push_back(1);
        list.push_back(2);
        list.push_back(3);

        let v: Vec<_> = list.into_iter().collect();
        assert_eq!(v, vec![1, 2, 3]);
    }
}
