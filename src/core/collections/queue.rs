use std::fmt::{Debug, Display};

/// 循环队列实现
///
/// 特点：
/// - 基于循环缓冲区实现
/// - 支持固定容量
/// - FIFO（先进先出）操作
/// - O(1) 时间复杂度的入队和出队操作
#[derive(Debug)]
pub struct Queue<T> {
    /// 存储数据的内部缓冲区
    buffer: Vec<Option<T>>,
    /// 队列头部位置
    head: usize,
    /// 队列尾部位置
    tail: usize,
    /// 当前队列中的元素数量
    size: usize,
}

impl<T> Queue<T> {
    /// 创建指定容量的新队列
    ///
    /// # 参数
    /// - `capacity`: 队列的初始容量
    ///
    /// # 返回
    /// 返回一个新的空队列
    pub fn with_capacity(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.extend((0..capacity).map(|_| None));
        Queue {
            buffer,
            head: 0,
            tail: 0,
            size: 0,
        }
    }

    /// 创建默认容量的新队列
    pub fn new() -> Self {
        Self::with_capacity(16)
    }

    /// 将元素添加到队列尾部
    ///
    /// # 参数
    /// - `value`: 要入队的元素
    ///
    /// # 返回
    /// 如果队列已满，返回 `Err(value)`，否则返回 `Ok(())`
    pub fn enqueue(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }

        self.buffer[self.tail] = Some(value);
        self.tail = (self.tail + 1) % self.capacity();
        self.size += 1;
        Ok(())
    }

    /// 从队列头部移除并返回元素
    ///
    /// # 返回
    /// 如果队列为空，返回 `None`，否则返回队首元素
    pub fn dequeue(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let value = self.buffer[self.head].take();
        self.head = (self.head + 1) % self.capacity();
        self.size -= 1;
        value
    }

    /// 查看队首元素但不移除
    ///
    /// # 返回
    /// 如果队列为空，返回 `None`，否则返回队首元素的引用
    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            self.buffer[self.head].as_ref()
        }
    }

    /// 返回队列当前大小
    pub fn len(&self) -> usize {
        self.size
    }

    /// 检查队列是否为空
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// 检查队列是否已满
    pub fn is_full(&self) -> bool {
        self.size == self.capacity()
    }

    /// 返回队列容量
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// 清空队列
    pub fn clear(&mut self) {
        for i in 0..self.capacity() {
            self.buffer[i] = None;
        }
        self.head = 0;
        self.tail = 0;
        self.size = 0;
    }
}

impl<T: Clone> Clone for Queue<T> {
    fn clone(&self) -> Self {
        Queue {
            buffer: self.buffer.clone(),
            head: self.head,
            tail: self.tail,
            size: self.size,
        }
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_queue() {
        let queue: Queue<i32> = Queue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert_eq!(queue.capacity(), 16);
    }

    #[test]
    fn test_enqueue_dequeue() {
        let mut queue = Queue::new();

        // 测试入队
        assert!(queue.enqueue(1).is_ok());
        assert!(queue.enqueue(2).is_ok());
        assert!(queue.enqueue(3).is_ok());
        assert_eq!(queue.len(), 3);

        // 测试出队
        assert_eq!(queue.dequeue(), Some(1));
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), None);
    }

    #[test]
    fn test_peek() {
        let mut queue = Queue::new();
        assert_eq!(queue.peek(), None);

        queue.enqueue(1).unwrap();
        assert_eq!(queue.peek(), Some(&1));

        queue.enqueue(2).unwrap();
        assert_eq!(queue.peek(), Some(&1));
    }

    #[test]
    fn test_clear() {
        let mut queue = Queue::new();
        queue.enqueue(1).unwrap();
        queue.enqueue(2).unwrap();

        queue.clear();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_full_queue() {
        let mut queue = Queue::with_capacity(3);

        assert!(queue.enqueue(1).is_ok());
        assert!(queue.enqueue(2).is_ok());
        assert!(queue.enqueue(3).is_ok());
        assert!(queue.enqueue(4).is_err());

        assert_eq!(queue.dequeue(), Some(1));
        assert!(queue.enqueue(4).is_ok());
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_circular_behavior() {
        let mut queue = Queue::with_capacity(3);

        // 填满队列
        assert!(queue.enqueue(1).is_ok());
        assert!(queue.enqueue(2).is_ok());
        assert!(queue.enqueue(3).is_ok());

        // 移除一个元素并添加新元素
        assert_eq!(queue.dequeue(), Some(1));
        assert!(queue.enqueue(4).is_ok());

        // 验证元素顺序
        assert_eq!(queue.dequeue(), Some(2));
        assert_eq!(queue.dequeue(), Some(3));
        assert_eq!(queue.dequeue(), Some(4));
    }
}
