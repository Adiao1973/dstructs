#![allow(unsafe_code)]
//! 二叉树实现
//!
//! 提供一个通用的二叉树实现，支持AVL平衡、父节点引用和自定义比较器

use std::cmp::Ordering;
use std::marker::PhantomData;
use std::mem;
use std::ptr::NonNull;

/// 二叉树节点
struct Node<T> {
    /// 节点数据
    data: T,
    /// 节点高度（用于AVL平衡）
    height: i32,
    /// 左子节点
    left: Option<NonNull<Node<T>>>,
    /// 右子节点
    right: Option<NonNull<Node<T>>>,
    /// 父节点（可选）
    parent: Option<NonNull<Node<T>>>,
}

/// 二叉树
pub struct BinaryTree<T, C = DefaultCompare>
where
    C: Comparator<T>,
{
    /// 根节点
    root: Option<NonNull<Node<T>>>,
    /// 树的大小
    size: usize,
    /// 比较器
    comparator: C,
    /// 生命周期标记
    marker: PhantomData<T>,
}

/// 默认比较器
#[derive(Default)]
pub struct DefaultCompare;

/// 比较器trait
pub trait Comparator<T> {
    /// 比较两个值
    fn compare(&self, a: &T, b: &T) -> Ordering;
}

/// 为实现Ord的类型实现默认比较器
impl<T: Ord> Comparator<T> for DefaultCompare {
    fn compare(&self, a: &T, b: &T) -> Ordering {
        a.cmp(b)
    }
}

impl<T> Node<T> {
    /// 创建新节点
    fn new(data: T) -> Self {
        Node {
            data,
            height: 1,
            left: None,
            right: None,
            parent: None,
        }
    }

    /// 分配新节点并返回裸指针
    fn into_ptr(self) -> NonNull<Self> {
        let node = Box::new(self);
        NonNull::new(Box::into_raw(node)).unwrap()
    }

    /// 获取节点高度
    fn height(&self) -> i32 {
        self.height
    }

    /// 获取平衡因子
    fn balance_factor(&self) -> i32 {
        let left_height = self
            .left
            .map_or(0, |left| unsafe { (*left.as_ptr()).height });
        let right_height = self
            .right
            .map_or(0, |right| unsafe { (*right.as_ptr()).height });
        left_height - right_height
    }

    /// 更新节点高度
    fn update_height(&mut self) {
        let left_height = self
            .left
            .map_or(0, |left| unsafe { (*left.as_ptr()).height });
        let right_height = self
            .right
            .map_or(0, |right| unsafe { (*right.as_ptr()).height });
        self.height = 1 + std::cmp::max(left_height, right_height);
    }
}

impl<T, C> BinaryTree<T, C>
where
    C: Comparator<T>,
{
    /// 创建新的二叉树
    pub fn new(comparator: C) -> Self {
        BinaryTree {
            root: None,
            size: 0,
            comparator,
            marker: PhantomData,
        }
    }

    /// 创建使用默认比较器的二叉树
    pub fn new_default() -> Self
    where
        C: Default,
    {
        Self::new(C::default())
    }

    /// 返回树的大小
    pub fn len(&self) -> usize {
        self.size
    }

    /// 判断树是否为空
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// 插入新值
    pub fn insert(&mut self, data: T) {
        let new_node = Node::new(data).into_ptr();
        if self.root.is_none() {
            self.root = Some(new_node);
            self.size = 1;
            return;
        }

        unsafe {
            let mut current = self.root;
            let mut parent = None;
            let mut is_left = false;

            // 找到插入位置
            while let Some(node) = current {
                parent = current;
                match self
                    .comparator
                    .compare(&(*new_node.as_ptr()).data, &(*node.as_ptr()).data)
                {
                    Ordering::Less => {
                        current = (*node.as_ptr()).left;
                        is_left = true;
                    }
                    Ordering::Greater => {
                        current = (*node.as_ptr()).right;
                        is_left = false;
                    }
                    Ordering::Equal => {
                        // 交换节点数据而不是移动
                        unsafe {
                            mem::swap(&mut (*node.as_ptr()).data, &mut (*new_node.as_ptr()).data);
                        }
                        // 释放新节点
                        let _ = unsafe { Box::from_raw(new_node.as_ptr()) };
                        return;
                    }
                }
            }

            // 设置父节点关系
            (*new_node.as_ptr()).parent = parent;
            if let Some(p) = parent {
                if is_left {
                    (*p.as_ptr()).left = Some(new_node);
                } else {
                    (*p.as_ptr()).right = Some(new_node);
                }
            }

            // 从插入点向上重新平衡树
            self.rebalance_after_insert(parent);
        }

        self.size += 1;
    }

    /// 查找值
    pub fn find(&self, data: &T) -> Option<&T> {
        unsafe {
            let mut current = self.root;
            while let Some(node) = current {
                match self.comparator.compare(data, &(*node.as_ptr()).data) {
                    Ordering::Less => current = (*node.as_ptr()).left,
                    Ordering::Greater => current = (*node.as_ptr()).right,
                    Ordering::Equal => return Some(&(*node.as_ptr()).data),
                }
            }
            None
        }
    }

    /// 在插入后重新平衡树
    unsafe fn rebalance_after_insert(&mut self, mut node: Option<NonNull<Node<T>>>) {
        while let Some(n) = node {
            (*n.as_ptr()).update_height();
            let balance = (*n.as_ptr()).balance_factor();

            if balance > 1 {
                let left = (*n.as_ptr()).left.unwrap();
                if (*left.as_ptr()).balance_factor() < 0 {
                    self.rotate_left(Some(left));
                }
                node = self.rotate_right(Some(n));
            } else if balance < -1 {
                let right = (*n.as_ptr()).right.unwrap();
                if (*right.as_ptr()).balance_factor() > 0 {
                    self.rotate_right(Some(right));
                }
                node = self.rotate_left(Some(n));
            }

            node = (*n.as_ptr()).parent;
        }
    }

    /// 左旋转
    unsafe fn rotate_left(&mut self, node: Option<NonNull<Node<T>>>) -> Option<NonNull<Node<T>>> {
        if let Some(n) = node {
            if let Some(right) = (*n.as_ptr()).right {
                (*n.as_ptr()).right = (*right.as_ptr()).left;
                if let Some(left) = (*right.as_ptr()).left {
                    (*left.as_ptr()).parent = Some(n);
                }
                (*right.as_ptr()).left = Some(n);
                (*right.as_ptr()).parent = (*n.as_ptr()).parent;
                (*n.as_ptr()).parent = Some(right);

                if let Some(parent) = (*right.as_ptr()).parent {
                    if (*parent.as_ptr()).left == Some(n) {
                        (*parent.as_ptr()).left = Some(right);
                    } else {
                        (*parent.as_ptr()).right = Some(right);
                    }
                } else {
                    self.root = Some(right);
                }

                (*n.as_ptr()).update_height();
                (*right.as_ptr()).update_height();

                return Some(right);
            }
        }
        node
    }

    /// 右旋转
    unsafe fn rotate_right(&mut self, node: Option<NonNull<Node<T>>>) -> Option<NonNull<Node<T>>> {
        if let Some(n) = node {
            if let Some(left) = (*n.as_ptr()).left {
                (*n.as_ptr()).left = (*left.as_ptr()).right;
                if let Some(right) = (*left.as_ptr()).right {
                    (*right.as_ptr()).parent = Some(n);
                }
                (*left.as_ptr()).right = Some(n);
                (*left.as_ptr()).parent = (*n.as_ptr()).parent;
                (*n.as_ptr()).parent = Some(left);

                if let Some(parent) = (*left.as_ptr()).parent {
                    if (*parent.as_ptr()).left == Some(n) {
                        (*parent.as_ptr()).left = Some(left);
                    } else {
                        (*parent.as_ptr()).right = Some(left);
                    }
                } else {
                    self.root = Some(left);
                }

                (*n.as_ptr()).update_height();
                (*left.as_ptr()).update_height();

                return Some(left);
            }
        }
        node
    }

    /// 删除节点
    pub fn remove(&mut self, data: &T) -> Option<T> {
        unsafe {
            let mut current = self.root;
            while let Some(node) = current {
                match self.comparator.compare(data, &(*node.as_ptr()).data) {
                    Ordering::Less => current = (*node.as_ptr()).left,
                    Ordering::Greater => current = (*node.as_ptr()).right,
                    Ordering::Equal => {
                        return Some(self.remove_node(node));
                    }
                }
            }
            None
        }
    }

    /// 内部删除节点实现
    unsafe fn remove_node(&mut self, node: NonNull<Node<T>>) -> T {
        let node = node.as_ptr();
        let parent = (*node).parent;

        match ((*node).left, (*node).right) {
            (None, None) => {
                // 叶子节点，直接删除
                self.update_parent_link(node, None);
                let node = Box::from_raw(node);
                self.size -= 1;
                self.rebalance_after_remove(parent);
                node.data
            }
            (Some(left), None) | (None, Some(left)) => {
                // 只有一个子节点
                let child = left;
                self.update_parent_link(node, Some(child));
                (*child.as_ptr()).parent = (*node).parent;
                let node = Box::from_raw(node);
                self.size -= 1;
                self.rebalance_after_remove(parent);
                node.data
            }
            (Some(_), Some(_)) => {
                // 有两个子节点，找到后继节点
                let successor = self.find_successor(node);
                mem::swap(&mut (*node).data, &mut (*successor.as_ptr()).data);
                self.remove_node(successor)
            }
        }
    }

    /// 更新父节点链接
    unsafe fn update_parent_link(
        &mut self,
        node: *mut Node<T>,
        new_child: Option<NonNull<Node<T>>>,
    ) {
        let parent = (*node).parent;
        match parent {
            None => self.root = new_child,
            Some(p) => {
                if (*p.as_ptr()).left == Some(NonNull::new(node).unwrap()) {
                    (*p.as_ptr()).left = new_child;
                } else {
                    (*p.as_ptr()).right = new_child;
                }
            }
        }
    }

    /// 找到节点的后继节点
    unsafe fn find_successor(&self, node: *mut Node<T>) -> NonNull<Node<T>> {
        let mut current = (*node).right.unwrap();
        while let Some(left) = (*current.as_ptr()).left {
            current = left;
        }
        current
    }

    /// 删除后重新平衡
    unsafe fn rebalance_after_remove(&mut self, mut node: Option<NonNull<Node<T>>>) {
        while let Some(n) = node {
            (*n.as_ptr()).update_height();
            let balance = (*n.as_ptr()).balance_factor();

            if balance > 1 {
                let left = (*n.as_ptr()).left.unwrap();
                if (*left.as_ptr()).balance_factor() < 0 {
                    self.rotate_left(Some(left));
                }
                node = self.rotate_right(Some(n));
            } else if balance < -1 {
                let right = (*n.as_ptr()).right.unwrap();
                if (*right.as_ptr()).balance_factor() > 0 {
                    self.rotate_right(Some(right));
                }
                node = self.rotate_left(Some(n));
            }

            node = (*n.as_ptr()).parent;
        }
    }

    /// 获取中序迭代器
    pub fn iter(&self) -> InorderIter<'_, T> {
        InorderIter::new(self)
    }

    /// 获取可变中序迭代器
    pub fn iter_mut(&mut self) -> InorderIterMut<'_, T> {
        InorderIterMut::new(self)
    }

    /// 获取前序迭代器
    pub fn iter_preorder(&self) -> PreorderIter<'_, T> {
        PreorderIter::new(self)
    }

    /// 获取后序迭代器
    pub fn iter_postorder(&self) -> PostorderIter<'_, T> {
        PostorderIter::new(self)
    }
}

/// 中序迭代器
pub struct InorderIter<'a, T> {
    stack: Vec<NonNull<Node<T>>>,
    current: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> InorderIter<'a, T> {
    fn new<C: Comparator<T>>(tree: &'a BinaryTree<T, C>) -> Self {
        let mut iter = InorderIter {
            stack: Vec::new(),
            current: tree.root,
            marker: PhantomData,
        };
        iter.push_left_edge();
        iter
    }

    fn push_left_edge(&mut self) {
        while let Some(node) = self.current {
            self.stack.push(node);
            unsafe {
                self.current = (*node.as_ptr()).left;
            }
        }
    }
}

impl<'a, T> Iterator for InorderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            unsafe {
                self.current = (*node.as_ptr()).right;
                self.push_left_edge();
                Some(&(*node.as_ptr()).data)
            }
        } else {
            None
        }
    }
}

/// 可变中序迭代器
pub struct InorderIterMut<'a, T> {
    stack: Vec<NonNull<Node<T>>>,
    current: Option<NonNull<Node<T>>>,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> InorderIterMut<'a, T> {
    fn new<C: Comparator<T>>(tree: &'a mut BinaryTree<T, C>) -> Self {
        let mut iter = InorderIterMut {
            stack: Vec::new(),
            current: tree.root,
            marker: PhantomData,
        };
        iter.push_left_edge();
        iter
    }

    fn push_left_edge(&mut self) {
        while let Some(node) = self.current {
            self.stack.push(node);
            unsafe {
                self.current = (*node.as_ptr()).left;
            }
        }
    }
}

impl<'a, T> Iterator for InorderIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            unsafe {
                self.current = (*node.as_ptr()).right;
                self.push_left_edge();
                Some(&mut (*node.as_ptr()).data)
            }
        } else {
            None
        }
    }
}

/// 前序迭代器
pub struct PreorderIter<'a, T> {
    stack: Vec<NonNull<Node<T>>>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> PreorderIter<'a, T> {
    fn new<C: Comparator<T>>(tree: &'a BinaryTree<T, C>) -> Self {
        let mut iter = PreorderIter {
            stack: Vec::new(),
            marker: PhantomData,
        };
        if let Some(root) = tree.root {
            iter.stack.push(root);
        }
        iter
    }
}

impl<'a, T> Iterator for PreorderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.stack.pop() {
            unsafe {
                if let Some(right) = (*node.as_ptr()).right {
                    self.stack.push(right);
                }
                if let Some(left) = (*node.as_ptr()).left {
                    self.stack.push(left);
                }
                Some(&(*node.as_ptr()).data)
            }
        } else {
            None
        }
    }
}

/// 后序迭代器
pub struct PostorderIter<'a, T> {
    stack: Vec<(NonNull<Node<T>>, bool)>,
    marker: PhantomData<&'a T>,
}

impl<'a, T> PostorderIter<'a, T> {
    fn new<C: Comparator<T>>(tree: &'a BinaryTree<T, C>) -> Self {
        let mut iter = PostorderIter {
            stack: Vec::new(),
            marker: PhantomData,
        };
        if let Some(root) = tree.root {
            iter.push_left_path(root);
        }
        iter
    }

    fn push_left_path(&mut self, mut node: NonNull<Node<T>>) {
        unsafe {
            loop {
                self.stack.push((node, false));
                if let Some(left) = (*node.as_ptr()).left {
                    node = left;
                } else {
                    break;
                }
            }
        }
    }
}

impl<'a, T> Iterator for PostorderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(&mut (node, ref mut visited)) = self.stack.last_mut() {
            unsafe {
                if !*visited {
                    *visited = true;
                    if let Some(right) = (*node.as_ptr()).right {
                        self.push_left_path(right);
                    }
                } else {
                    self.stack.pop();
                    return Some(&(*node.as_ptr()).data);
                }
            }
        }
        None
    }
}

// 为 BinaryTree 实现 IntoIterator
impl<'a, T, C> IntoIterator for &'a BinaryTree<T, C>
where
    C: Comparator<T>,
{
    type Item = &'a T;
    type IntoIter = InorderIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, C> IntoIterator for &'a mut BinaryTree<T, C>
where
    C: Comparator<T>,
{
    type Item = &'a mut T;
    type IntoIter = InorderIterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_find() {
        let mut tree = BinaryTree::<i32>::new_default();
        assert!(tree.is_empty());

        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);

        assert_eq!(tree.len(), 5);
        assert!(!tree.is_empty());

        assert!(tree.find(&5).is_some());
        assert!(tree.find(&3).is_some());
        assert!(tree.find(&7).is_some());
        assert!(tree.find(&1).is_some());
        assert!(tree.find(&9).is_some());
        assert!(tree.find(&4).is_none());
    }

    #[test]
    fn test_balance() {
        let mut tree = BinaryTree::<i32>::new_default();

        // 插入有序数据，测试平衡
        for i in 1..=7 {
            tree.insert(i);
        }

        unsafe {
            // 验证根节点的平衡因子
            let root = tree.root.unwrap();
            assert!((*root.as_ptr()).balance_factor().abs() <= 1);

            // 验证所有节点的平衡因子
            fn check_balance<T>(node: Option<NonNull<Node<T>>>) -> bool {
                match node {
                    None => true,
                    Some(n) => unsafe {
                        let balance = (*n.as_ptr()).balance_factor();
                        balance.abs() <= 1
                            && check_balance((*n.as_ptr()).left)
                            && check_balance((*n.as_ptr()).right)
                    },
                }
            }

            assert!(check_balance(Some(root)));
        }
    }

    #[test]
    fn test_custom_comparator() {
        struct ReverseCompare;
        impl Comparator<i32> for ReverseCompare {
            fn compare(&self, a: &i32, b: &i32) -> Ordering {
                b.cmp(a)
            }
        }

        let mut tree = BinaryTree::new(ReverseCompare);
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);

        assert!(tree.find(&5).is_some());
        assert!(tree.find(&3).is_some());
        assert!(tree.find(&7).is_some());
    }

    #[test]
    fn test_remove() {
        let mut tree = BinaryTree::<i32>::new_default();
        tree.insert(5);
        tree.insert(3);
        tree.insert(7);
        tree.insert(1);
        tree.insert(9);

        assert_eq!(tree.remove(&3), Some(3));
        assert_eq!(tree.len(), 4);
        assert!(tree.find(&3).is_none());

        assert_eq!(tree.remove(&5), Some(5));
        assert_eq!(tree.len(), 3);
        assert!(tree.find(&5).is_none());

        assert_eq!(tree.remove(&10), None);
    }

    #[test]
    fn test_iterators() {
        let mut tree = BinaryTree::<i32>::new_default();
        for i in &[5, 3, 7, 1, 9] {
            tree.insert(*i);
        }

        // 测试中序遍历
        let inorder: Vec<_> = tree.iter().copied().collect();
        assert_eq!(inorder, vec![1, 3, 5, 7, 9]);

        // 测试前序遍历
        let preorder: Vec<_> = tree.iter_preorder().copied().collect();
        assert_eq!(preorder, vec![5, 3, 1, 7, 9]);

        // 测试后序遍历
        let postorder: Vec<_> = tree.iter_postorder().copied().collect();
        assert_eq!(postorder, vec![1, 3, 9, 7, 5]);

        // 测试可变迭代器
        for x in &mut tree {
            *x += 10;
        }
        let modified: Vec<_> = tree.iter().copied().collect();
        assert_eq!(modified, vec![11, 13, 15, 17, 19]);
    }
}
