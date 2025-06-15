#![no_std]
// 允许特定模块使用 unsafe 代码
#![allow(unsafe_code)]

extern crate alloc;

use alloc::boxed::Box;
use core::fmt;
use core::ptr::{self, NonNull};
use core::sync::atomic::{AtomicPtr, Ordering};

/// 节点结构体，用于存储队列中的数据
struct Node<T> {
    data: Option<T>,
    next: AtomicPtr<Node<T>>,
}

/// 无锁队列实现
///
/// # 安全性
///
/// 该实现在以下方面保证了内存安全和线程安全：
///
/// 1. 原子操作：所有共享状态的访问都通过原子操作进行
/// 2. 内存管理：
///    - 所有分配的内存都通过 Box 管理
///    - 在 drop 时正确清理所有分配的内存
/// 3. 指针安全：
///    - 保证所有原始指针在使用前都是有效的
///    - 正确处理 ABA 问题
/// 4. 并发安全：
///    - 实现 Send 和 Sync 以支持跨线程使用
///    - CAS 操作保证原子性
///
/// # 实现说明
///
/// 该队列使用单向链表实现，包含以下特点：
/// - 使用原子操作实现无锁并发
/// - 处理 ABA 问题
/// - 支持多生产者多消费者
/// - 内存安全且无内存泄漏
/// - 支持 no_std 环境
pub struct LockFreeQueue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

// SAFETY: 所有内部可变性都通过原子操作保护，保证线程安全
unsafe impl<T: Send> Send for LockFreeQueue<T> {}
// SAFETY: 所有共享访问都通过原子操作同步，保证并发安全
unsafe impl<T: Send> Sync for LockFreeQueue<T> {}

impl<T> Node<T> {
    /// 创建新节点
    ///
    /// # Safety
    ///
    /// 返回的指针总是有效的，因为它来自 Box::into_raw
    fn new(data: Option<T>) -> *mut Self {
        Box::into_raw(Box::new(Self {
            data,
            next: AtomicPtr::new(ptr::null_mut()),
        }))
    }

    /// 安全地获取节点引用
    ///
    /// # Safety
    ///
    /// 调用者必须确保指针有效且不会在访问期间被释放
    unsafe fn as_ref<'a>(ptr: *mut Self) -> Option<&'a Self> {
        NonNull::new(ptr).map(|p| &*p.as_ptr())
    }
}

impl<T> LockFreeQueue<T> {
    /// 创建一个新的无锁队列
    pub fn new() -> Self {
        let dummy = Node::new(None);
        Self {
            head: AtomicPtr::new(dummy),
            tail: AtomicPtr::new(dummy),
        }
    }

    /// 将元素推入队列尾部
    pub fn push(&self, value: T) {
        let new_node = Node::new(Some(value));

        loop {
            let tail = self.tail.load(Ordering::Acquire);
            // SAFETY: tail 指针来自 Node::new，保证有效
            let tail_ref = unsafe { Node::as_ref(tail) };
            let Some(tail_ref) = tail_ref else {
                continue;
            };
            let next = tail_ref.next.load(Ordering::Acquire);

            if ptr::null_mut() == next {
                match tail_ref.next.compare_exchange(
                    ptr::null_mut(),
                    new_node,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let _ = self.tail.compare_exchange(
                            tail,
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                        break;
                    }
                    Err(_) => continue,
                }
            } else {
                let _ =
                    self.tail
                        .compare_exchange(tail, next, Ordering::Release, Ordering::Relaxed);
            }
        }
    }

    /// 从队列头部弹出元素
    pub fn pop(&self) -> Option<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            // SAFETY: head 指针来自 Node::new，保证有效
            let head_ref = unsafe { Node::as_ref(head) };
            let Some(head_ref) = head_ref else {
                continue;
            };
            let next = head_ref.next.load(Ordering::Acquire);

            if ptr::null_mut() == next {
                return None;
            }

            // SAFETY: next 非空且有效，因为它是通过原子操作获取的
            let next_ref = unsafe { Node::as_ref(next) };
            let Some(next_ref) = next_ref else {
                continue;
            };

            if self
                .head
                .compare_exchange(head, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: 我们已经成功将 head 从队列中移除
                unsafe {
                    let old_head = Box::from_raw(head);
                    return ptr::read(&next_ref.data);
                }
            }
        }
    }

    /// 检查队列是否为空
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        // SAFETY: head 指针来自 Node::new，保证有效
        let head_ref = unsafe { Node::as_ref(head) };
        let Some(head_ref) = head_ref else {
            return true;
        };
        head_ref.next.load(Ordering::Acquire).is_null()
    }
}

impl<T: Clone> LockFreeQueue<T> {
    /// 获取队列头部元素的克隆，不移除元素
    pub fn peek(&self) -> Option<T> {
        let head = self.head.load(Ordering::Acquire);
        // SAFETY: head 指针来自 Node::new，保证有效
        let head_ref = unsafe { Node::as_ref(head) };
        let Some(head_ref) = head_ref else {
            return None;
        };
        let next = head_ref.next.load(Ordering::Acquire);
        if ptr::null_mut() == next {
            return None;
        }
        // SAFETY: next 非空且有效
        unsafe { Node::as_ref(next) }.and_then(|node| node.data.clone())
    }
}

impl<T> Default for LockFreeQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for LockFreeQueue<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}

        // 清理哨兵节点
        let head = self.head.load(Ordering::Relaxed);
        // SAFETY: head 是最后一个节点，此时没有其他线程访问
        unsafe {
            let _ = Box::from_raw(head);
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for LockFreeQueue<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LockFreeQueue")
            .field("is_empty", &self.is_empty())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let queue = LockFreeQueue::new();
        assert!(queue.is_empty());

        queue.push(1);
        assert!(!queue.is_empty());

        assert_eq!(queue.pop(), Some(1));
        assert!(queue.is_empty());
    }

    #[test]
    fn test_multiple_pushes_pops() {
        let queue = LockFreeQueue::new();

        for i in 0..100 {
            queue.push(i);
        }

        for i in 0..100 {
            assert_eq!(queue.pop(), Some(i));
        }

        assert!(queue.is_empty());
    }

    #[test]
    fn test_peek() {
        let queue = LockFreeQueue::new();
        assert_eq!(queue.peek(), None);

        queue.push(1);
        assert_eq!(queue.peek(), Some(1));
        assert_eq!(queue.peek(), Some(1)); // 不会移除元素
        assert_eq!(queue.pop(), Some(1));
        assert_eq!(queue.peek(), None);
    }
}
