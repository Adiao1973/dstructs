//! 二叉堆实现
//!
//! 提供最大堆和最小堆实现，支持优先级队列操作和固定容量选项

use std::cmp::Ordering;

/// 优先级队列元素的特征
pub trait PriorityElement {
    /// 优先级的类型
    type Priority: Ord;

    /// 获取元素的优先级
    fn priority(&self) -> Self::Priority;
}

/// 优先级队列的特征
pub trait PriorityQueue<T: PriorityElement> {
    /// 插入元素
    fn enqueue(&mut self, element: T) -> Result<(), T>;
    /// 获取最高优先级元素
    fn peek(&self) -> Option<&T>;
    /// 移除并返回最高优先级元素
    fn dequeue(&mut self) -> Option<T>;
    /// 当前队列大小
    fn len(&self) -> usize;
    /// 检查队列是否为空
    fn is_empty(&self) -> bool;
}

/// 堆的类型（最大堆或最小堆）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapType {
    /// 最大堆：父节点大于等于子节点
    MaxHeap,
    /// 最小堆：父节点小于等于子节点
    MinHeap,
}

/// 二叉堆的元素包装器
#[derive(Debug)]
struct HeapEntry<T> {
    /// 存储的数据
    data: T,
    /// 在堆中的索引
    index: usize,
}

/// 二叉堆
pub struct BinaryHeap<T> {
    /// 存储堆元素的向量
    data: Vec<HeapEntry<T>>,
    /// 堆的类型
    heap_type: HeapType,
    /// 堆的容量限制（如果有）
    capacity: Option<usize>,
}

impl<T: Ord> BinaryHeap<T> {
    /// 创建新的二叉堆
    pub fn new(heap_type: HeapType) -> Self {
        BinaryHeap {
            data: Vec::new(),
            heap_type,
            capacity: None,
        }
    }

    /// 创建具有固定容量的二叉堆
    pub fn with_capacity(heap_type: HeapType, capacity: usize) -> Self {
        BinaryHeap {
            data: Vec::with_capacity(capacity),
            heap_type,
            capacity: Some(capacity),
        }
    }

    /// 返回堆中的元素数量
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 判断堆是否为空
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 判断堆是否已满（对于有容量限制的堆）
    pub fn is_full(&self) -> bool {
        if let Some(capacity) = self.capacity {
            self.len() >= capacity
        } else {
            false
        }
    }

    /// 获取堆顶元素的引用
    pub fn peek(&self) -> Option<&T> {
        self.data.first().map(|entry| &entry.data)
    }

    /// 获取堆顶元素的可变引用
    pub fn peek_mut(&mut self) -> Option<&mut T> {
        self.data.first_mut().map(|entry| &mut entry.data)
    }

    /// 插入新元素
    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }

        self.data.push(HeapEntry {
            data: value,
            index: self.len(),
        });
        self.sift_up(self.len() - 1);
        Ok(())
    }

    /// 移除并返回堆顶元素
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }

        let value = self.data.swap_remove(0);
        if !self.is_empty() {
            self.sift_down(0);
        }
        Some(value.data)
    }

    /// 清空堆
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// 将向量转换为堆
    pub fn from_vec(mut vec: Vec<T>, heap_type: HeapType) -> Self {
        let mut heap = BinaryHeap {
            data: vec
                .into_iter()
                .map(|value| HeapEntry {
                    data: value,
                    index: 0,
                })
                .collect(),
            heap_type,
            capacity: None,
        };
        heap.heapify();
        heap
    }

    /// 获取堆的类型
    pub fn heap_type(&self) -> HeapType {
        self.heap_type
    }

    /// 比较两个元素（考虑堆类型）
    fn compare(&self, a: &T, b: &T) -> Ordering {
        match self.heap_type {
            HeapType::MaxHeap => a.cmp(b),
            HeapType::MinHeap => b.cmp(a),
        }
    }

    /// 获取父节点索引
    fn parent(index: usize) -> Option<usize> {
        if index > 0 {
            Some((index - 1) / 2)
        } else {
            None
        }
    }

    /// 获取左子节点索引
    fn left_child(index: usize) -> usize {
        2 * index + 1
    }

    /// 获取右子节点索引
    fn right_child(index: usize) -> usize {
        2 * index + 2
    }

    /// 向上调整堆
    fn sift_up(&mut self, mut index: usize) {
        while let Some(parent) = Self::parent(index) {
            if self
                .compare(&self.data[index].data, &self.data[parent].data)
                .is_gt()
            {
                self.data.swap(index, parent);
                // 更新索引
                self.data[index].index = index;
                self.data[parent].index = parent;
                index = parent;
            } else {
                break;
            }
        }
    }

    /// 向下调整堆
    fn sift_down(&mut self, mut index: usize) {
        let len = self.len();
        loop {
            let left = Self::left_child(index);
            let right = Self::right_child(index);
            let mut largest = index;

            if left < len
                && self
                    .compare(&self.data[left].data, &self.data[largest].data)
                    .is_gt()
            {
                largest = left;
            }
            if right < len
                && self
                    .compare(&self.data[right].data, &self.data[largest].data)
                    .is_gt()
            {
                largest = right;
            }

            if largest == index {
                break;
            }

            self.data.swap(index, largest);
            // 更新索引
            self.data[index].index = index;
            self.data[largest].index = largest;
            index = largest;
        }
    }

    /// 将无序向量转换为堆
    fn heapify(&mut self) {
        if self.len() <= 1 {
            return;
        }

        // 从最后一个非叶子节点开始向下调整
        let last_parent = Self::parent(self.len() - 1).unwrap();
        for i in (0..=last_parent).rev() {
            self.sift_down(i);
        }
    }

    /// 减小指定索引处元素的值
    ///
    /// # 参数
    /// * `index` - 要修改的元素索引
    /// * `new_value` - 新的值（必须小于当前值）
    ///
    /// # 返回
    /// * `Ok(())` - 操作成功
    /// * `Err(())` - 操作失败（索引无效或新值不小于当前值）
    pub fn decrease_key(&mut self, index: usize, new_value: T) -> Result<(), ()> {
        if index >= self.len() {
            return Err(());
        }

        // 对于最小堆，新值必须小于当前值
        // 对于最大堆，新值必须大于当前值
        let is_valid = match self.heap_type {
            HeapType::MinHeap => new_value < self.data[index].data,
            HeapType::MaxHeap => new_value > self.data[index].data,
        };

        if !is_valid {
            return Err(());
        }

        self.data[index].data = new_value;
        self.sift_up(index);
        Ok(())
    }

    /// 增加指定索引处元素的值
    ///
    /// # 参数
    /// * `index` - 要修改的元素索引
    /// * `new_value` - 新的值
    ///
    /// # 返回
    /// * `Ok(())` - 如果更新成功
    /// * `Err(())` - 如果更新失败（例如，索引无效或新值不满足堆的性质）
    pub fn increase_key(&mut self, index: usize, new_value: T) -> Result<(), ()> {
        if index >= self.len() {
            return Err(());
        }

        // 对于最大堆，新值必须大于当前值
        // 对于最小堆，新值必须小于当前值
        if self.compare(&new_value, &self.data[index].data).is_lt() {
            return Err(());
        }

        self.data[index].data = new_value;
        self.sift_up(index);
        Ok(())
    }

    /// 更新指定索引处元素的值
    pub fn update_key(&mut self, index: usize, new_value: T) -> Result<(), ()> {
        if index >= self.len() {
            return Err(());
        }

        match self.compare(&new_value, &self.data[index].data) {
            Ordering::Less => self.decrease_key(index, new_value),
            Ordering::Greater => self.increase_key(index, new_value),
            Ordering::Equal => Ok(()),
        }
    }

    /// 删除指定索引处的元素
    ///
    /// # 参数
    /// * `index` - 要删除的元素索引
    ///
    /// # 返回
    /// 返回被删除的元素，如果索引无效则返回 None
    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index >= self.len() {
            return None;
        }

        let len = self.len();
        // 如果要删除的是最后一个元素，直接弹出
        if index == len - 1 {
            return self.data.pop().map(|entry| entry.data);
        }

        // 将最后一个元素移到要删除的位置
        self.data.swap(index, len - 1);
        let removed = self.data.pop().map(|entry| entry.data);

        if index < self.len() {
            // 获取父节点（如果存在）
            let parent_idx = Self::parent(index);

            // 检查与父节点的关系
            if let Some(parent) = parent_idx {
                let parent_relation = match self.heap_type {
                    HeapType::MinHeap => self.data[index].data < self.data[parent].data,
                    HeapType::MaxHeap => self.data[index].data > self.data[parent].data,
                };

                if parent_relation {
                    // 如果违反了与父节点的关系，需要上浮
                    self.sift_up(index);
                    return removed;
                }
            }

            // 如果不需要上浮，检查是否需要下沉
            let left = Self::left_child(index);
            let right = Self::right_child(index);
            let mut need_sift_down = false;

            if left < self.len() {
                let left_relation = match self.heap_type {
                    HeapType::MinHeap => self.data[left].data < self.data[index].data,
                    HeapType::MaxHeap => self.data[left].data > self.data[index].data,
                };
                if left_relation {
                    need_sift_down = true;
                }
            }

            if right < self.len() && !need_sift_down {
                let right_relation = match self.heap_type {
                    HeapType::MinHeap => self.data[right].data < self.data[index].data,
                    HeapType::MaxHeap => self.data[right].data > self.data[index].data,
                };
                if right_relation {
                    need_sift_down = true;
                }
            }

            if need_sift_down {
                self.sift_down(index);
            }
        }

        removed
    }

    /// 获取指定索引处元素的不可变引用
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index).map(|entry| &entry.data)
    }

    /// 获取指定索引处元素的可变引用
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index).map(|entry| &mut entry.data)
    }

    /// 合并另一个堆到当前堆
    pub fn merge(&mut self, other: Self) -> Result<(), Self> {
        if let Some(cap) = self.capacity {
            if self.len() + other.len() > cap {
                return Err(other);
            }
        }

        for entry in other.data {
            if let Err(_) = self.push(entry.data) {
                break;
            }
        }
        Ok(())
    }

    /// 验证堆的性质是否满足
    fn is_valid_heap(&self) -> bool {
        for i in 0..self.len() {
            let left = Self::left_child(i);
            let right = Self::right_child(i);

            if left < self.len() {
                if self
                    .compare(&self.data[left].data, &self.data[i].data)
                    .is_gt()
                {
                    return false;
                }
            }

            if right < self.len() {
                if self
                    .compare(&self.data[right].data, &self.data[i].data)
                    .is_gt()
                {
                    return false;
                }
            }
        }
        true
    }
}

// 实现优先级队列特征
impl<T: Ord + PriorityElement> PriorityQueue<T> for BinaryHeap<T> {
    fn enqueue(&mut self, element: T) -> Result<(), T> {
        self.push(element)
    }

    fn peek(&self) -> Option<&T> {
        self.peek()
    }

    fn dequeue(&mut self) -> Option<T> {
        self.pop()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }
}

impl<T: Ord> FromIterator<T> for BinaryHeap<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();
        Self::from_vec(vec, HeapType::MaxHeap)
    }
}

impl<T: Ord> Extend<T> for BinaryHeap<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            let _ = self.push(value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_heap() {
        let mut heap = BinaryHeap::new(HeapType::MaxHeap);

        // 测试插入
        heap.push(3).unwrap();
        heap.push(1).unwrap();
        heap.push(4).unwrap();
        heap.push(2).unwrap();

        // 测试弹出（应该按降序）
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_min_heap() {
        let mut heap = BinaryHeap::new(HeapType::MinHeap);

        // 测试插入
        heap.push(3).unwrap();
        heap.push(1).unwrap();
        heap.push(4).unwrap();
        heap.push(2).unwrap();

        // 测试弹出（应该按升序）
        assert_eq!(heap.pop(), Some(1));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_capacity() {
        let mut heap = BinaryHeap::with_capacity(HeapType::MaxHeap, 3);

        // 测试容量限制
        assert!(heap.push(1).is_ok());
        assert!(heap.push(2).is_ok());
        assert!(heap.push(3).is_ok());
        assert!(heap.push(4).is_err());

        // 测试弹出后可以继续插入
        heap.pop();
        assert!(heap.push(4).is_ok());
    }

    #[test]
    fn test_from_vec() {
        let vec = vec![3, 1, 4, 2];
        let mut heap = BinaryHeap::from_vec(vec, HeapType::MaxHeap);

        // 验证堆属性
        assert_eq!(heap.pop(), Some(4));
        assert_eq!(heap.pop(), Some(3));
        assert_eq!(heap.pop(), Some(2));
        assert_eq!(heap.pop(), Some(1));
    }

    #[test]
    fn test_peek() {
        let mut heap = BinaryHeap::new(HeapType::MaxHeap);

        assert_eq!(heap.peek(), None);

        heap.push(1).unwrap();
        heap.push(2).unwrap();

        assert_eq!(heap.peek(), Some(&2));

        // 测试peek_mut
        if let Some(value) = heap.peek_mut() {
            *value = 3;
        }

        assert_eq!(heap.pop(), Some(3));
    }

    #[test]
    fn test_clear() {
        let mut heap = BinaryHeap::new(HeapType::MaxHeap);
        heap.push(1).unwrap();
        heap.push(2).unwrap();

        heap.clear();
        assert!(heap.is_empty());
        assert_eq!(heap.pop(), None);
    }

    #[test]
    fn test_from_iterator() {
        let heap: BinaryHeap<i32> = vec![3, 1, 4, 2].into_iter().collect();

        assert_eq!(heap.len(), 4);
        assert_eq!(heap.peek(), Some(&4));
    }

    #[test]
    fn test_extend() {
        let mut heap = BinaryHeap::new(HeapType::MaxHeap);
        heap.push(1).unwrap();

        heap.extend(vec![3, 2, 4]);

        assert_eq!(heap.len(), 4);
        assert_eq!(heap.peek(), Some(&4));
    }

    #[test]
    fn test_decrease_key() {
        let mut heap = BinaryHeap::new(HeapType::MinHeap);
        heap.push(5).unwrap();
        heap.push(3).unwrap();
        heap.push(7).unwrap();

        assert!(heap.decrease_key(0, 1).is_ok());
        assert_eq!(heap.peek(), Some(&1));

        // 尝试将值增加（应该失败）
        assert!(heap.decrease_key(0, 6).is_err());
    }

    #[test]
    fn test_increase_key() {
        let mut heap = BinaryHeap::new(HeapType::MaxHeap);
        heap.push(5).unwrap();
        heap.push(3).unwrap();
        heap.push(7).unwrap();

        assert!(heap.increase_key(1, 8).is_ok());
        assert_eq!(heap.peek(), Some(&8));

        // 尝试将值减小（应该失败）
        assert!(heap.increase_key(0, 4).is_err());
    }

    #[test]
    fn test_remove() {
        let mut heap = BinaryHeap::new(HeapType::MinHeap);
        heap.push(5).unwrap();
        heap.push(3).unwrap();
        heap.push(7).unwrap();

        assert_eq!(heap.remove(1), Some(5));
        assert_eq!(heap.len(), 2);
        assert!(heap.is_valid_heap());
    }

    #[test]
    fn test_merge() {
        let mut heap1 = BinaryHeap::new(HeapType::MinHeap);
        heap1.push(1).unwrap();
        heap1.push(3).unwrap();

        let mut heap2 = BinaryHeap::new(HeapType::MinHeap);
        heap2.push(2).unwrap();
        heap2.push(4).unwrap();

        assert!(heap1.merge(heap2).is_ok());
        assert_eq!(heap1.len(), 4);
        assert_eq!(heap1.pop(), Some(1));
        assert_eq!(heap1.pop(), Some(2));
        assert_eq!(heap1.pop(), Some(3));
        assert_eq!(heap1.pop(), Some(4));
    }

    // 添加优先级队列测试
    #[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
    struct Task {
        priority: i32,
        id: String,
    }

    impl PriorityElement for Task {
        type Priority = i32;

        fn priority(&self) -> Self::Priority {
            self.priority
        }
    }

    #[test]
    fn test_priority_queue() {
        let mut pq: BinaryHeap<Task> = BinaryHeap::new(HeapType::MaxHeap);

        let task1 = Task {
            priority: 1,
            id: "low".to_string(),
        };
        let task2 = Task {
            priority: 3,
            id: "high".to_string(),
        };
        let task3 = Task {
            priority: 2,
            id: "medium".to_string(),
        };

        pq.enqueue(task1).unwrap();
        pq.enqueue(task2).unwrap();
        pq.enqueue(task3).unwrap();

        assert_eq!(pq.dequeue().unwrap().priority, 3);
        assert_eq!(pq.dequeue().unwrap().priority, 2);
        assert_eq!(pq.dequeue().unwrap().priority, 1);
        assert!(pq.is_empty());
    }
}
