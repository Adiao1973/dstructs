//! 哈希映射实现
//!
//! 提供高效的哈希映射实现，使用开放寻址法和线性探测解决冲突

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

/// 哈希映射中的条目状态
#[derive(Debug, Clone, PartialEq)]
enum EntryState {
    /// 空条目
    Empty,
    /// 已删除的条目
    Deleted,
    /// 已占用的条目
    Occupied,
}

/// 哈希映射中的条目
#[derive(Debug, Clone)]
struct Entry<K, V> {
    /// 键
    key: Option<K>,
    /// 值
    value: Option<V>,
    /// 条目状态
    state: EntryState,
}

impl<K, V> Entry<K, V> {
    /// 创建新的空条目
    fn new_empty() -> Self {
        Entry {
            key: None,
            value: None,
            state: EntryState::Empty,
        }
    }

    /// 创建新的已删除条目
    fn new_deleted() -> Self {
        Entry {
            key: None,
            value: None,
            state: EntryState::Deleted,
        }
    }

    /// 创建新的已占用条目
    fn new_occupied(key: K, value: V) -> Self {
        Entry {
            key: Some(key),
            value: Some(value),
            state: EntryState::Occupied,
        }
    }
}

/// 哈希映射
pub struct HashMap<K, V, S = RandomState> {
    /// 存储条目的向量
    entries: Vec<Entry<K, V>>,
    /// 实际元素数量
    size: usize,
    /// 已删除元素数量
    deleted: usize,
    /// 哈希生成器
    hash_builder: S,
    /// 负载因子阈值（默认0.75）
    load_factor_threshold: f64,
    /// 标记类型参数的生命周期
    marker: PhantomData<(K, V)>,
}

impl<K: Hash + Eq, V> HashMap<K, V, RandomState> {
    /// 创建新的哈希映射
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(0, RandomState::new())
    }

    /// 使用指定容量创建哈希映射
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// 使用指定容量和哈希生成器创建哈希映射
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> Self {
        let actual_capacity = if capacity == 0 {
            0
        } else {
            capacity.next_power_of_two()
        };

        let mut entries = Vec::with_capacity(actual_capacity);
        for _ in 0..actual_capacity {
            entries.push(Entry::new_empty());
        }

        HashMap {
            entries,
            size: 0,
            deleted: 0,
            hash_builder,
            load_factor_threshold: 0.75,
            marker: PhantomData,
        }
    }

    /// 返回映射中的元素数量
    pub fn len(&self) -> usize {
        self.size
    }

    /// 判断映射是否为空
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// 清空映射
    pub fn clear(&mut self) {
        self.entries.clear();
        self.size = 0;
        self.deleted = 0;
    }

    /// 计算键的哈希值
    fn hash<Q: ?Sized>(&self, key: &Q) -> u64
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let mut hasher = self.hash_builder.build_hasher();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// 查找键对应的位置
    fn find_slot<Q: ?Sized>(&self, key: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        if self.entries.is_empty() {
            return None;
        }

        let hash = self.hash(key);
        let mask = self.entries.len() - 1;
        let mut index = hash as usize & mask;
        let mut first_deleted = None;

        loop {
            match self.entries[index].state {
                EntryState::Empty => {
                    return first_deleted.or(Some(index));
                }
                EntryState::Deleted => {
                    if first_deleted.is_none() {
                        first_deleted = Some(index);
                    }
                }
                EntryState::Occupied => {
                    if let Some(ref k) = self.entries[index].key {
                        if key == k.borrow() {
                            return Some(index);
                        }
                    }
                }
            }

            index = (index + 1) & mask;
        }
    }

    /// 检查是否需要扩容
    fn needs_rehash(&self) -> bool {
        if self.entries.is_empty() {
            return self.size > 0;
        }

        let load_factor = (self.size + self.deleted) as f64 / self.entries.len() as f64;
        load_factor >= self.load_factor_threshold
    }

    /// 重新哈希
    fn rehash(&mut self) {
        let old_entries = mem::replace(&mut self.entries, Vec::new());
        let new_capacity = if self.size == 0 {
            0
        } else {
            (old_entries.len() * 2).max(16)
        };

        self.entries = Vec::with_capacity(new_capacity);
        for _ in 0..new_capacity {
            self.entries.push(Entry::new_empty());
        }

        self.deleted = 0;

        for entry in old_entries {
            if entry.state == EntryState::Occupied {
                if let Some(ref key) = entry.key {
                    let index = self.find_slot(key).unwrap();
                    self.entries[index] = entry;
                }
            }
        }
    }

    /// 获取键对应的值的引用
    pub fn get<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let index = self.find_slot(key)?;
        match self.entries[index].state {
            EntryState::Occupied => self.entries[index].value.as_ref(),
            _ => None,
        }
    }

    /// 获取键对应的值的可变引用
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let index = self.find_slot(key)?;
        match self.entries[index].state {
            EntryState::Occupied => self.entries[index].value.as_mut(),
            _ => None,
        }
    }

    /// 插入键值对
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if self.needs_rehash() {
            self.rehash();
        }

        if self.entries.is_empty() {
            self.entries.resize_with(16, Entry::new_empty);
        }

        let index = self.find_slot(&key).unwrap();
        match self.entries[index].state {
            EntryState::Occupied => {
                let old_value = self.entries[index].value.replace(value);
                old_value
            }
            EntryState::Deleted => {
                self.entries[index] = Entry::new_occupied(key, value);
                self.deleted -= 1;
                self.size += 1;
                None
            }
            EntryState::Empty => {
                self.entries[index] = Entry::new_occupied(key, value);
                self.size += 1;
                None
            }
        }
    }

    /// 删除键对应的值
    pub fn remove<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let index = self.find_slot(key)?;
        match self.entries[index].state {
            EntryState::Occupied => {
                let entry = &mut self.entries[index];
                entry.state = EntryState::Deleted;
                self.size -= 1;
                self.deleted += 1;
                entry.value.take()
            }
            _ => None,
        }
    }

    /// 检查是否包含指定的键
    pub fn contains_key<Q: ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.get(key).is_some()
    }

    /// 获取迭代器
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            entries: &self.entries,
            index: 0,
        }
    }

    /// 获取可变迭代器
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            entries: &mut self.entries,
            index: 0,
        }
    }
}

/// 迭代器
pub struct Iter<'a, K, V> {
    entries: &'a [Entry<K, V>],
    index: usize,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.entries.len() {
            let entry = &self.entries[self.index];
            self.index += 1;
            if entry.state == EntryState::Occupied {
                return Some((entry.key.as_ref().unwrap(), entry.value.as_ref().unwrap()));
            }
        }
        None
    }
}

/// 可变迭代器
pub struct IterMut<'a, K, V> {
    entries: &'a mut [Entry<K, V>],
    index: usize,
}

#[allow(unsafe_code)]
impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.entries.len() {
            let entry = &mut self.entries[self.index];
            self.index += 1;
            if entry.state == EntryState::Occupied {
                // SAFETY: 以下操作是安全的，因为：
                // 1. key 和 value 在 Occupied 状态下一定是有效的
                // 2. 我们通过裸指针来处理生命周期，避免借用检查器的限制
                // 3. 返回的引用不会超过 entry 的生命周期
                // 4. 每次迭代只会返回不同的元素，不会有重叠的可变引用
                let key = entry.key.as_ref().unwrap() as *const K;
                let value = entry.value.as_mut().unwrap() as *mut V;
                return Some(unsafe { (&*key, &mut *value) });
            }
        }
        None
    }
}

impl<K: Clone, V: Clone, S: Clone> Clone for HashMap<K, V, S> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
            size: self.size,
            deleted: self.deleted,
            hash_builder: self.hash_builder.clone(),
            load_factor_threshold: self.load_factor_threshold,
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut map = HashMap::new();
        assert!(map.is_empty());

        // 测试插入
        assert_eq!(map.insert(1, "one"), None);
        assert_eq!(map.insert(2, "two"), None);
        assert_eq!(map.len(), 2);

        // 测试获取
        assert_eq!(map.get(&1), Some(&"one"));
        assert_eq!(map.get(&2), Some(&"two"));
        assert_eq!(map.get(&3), None);

        // 测试更新
        assert_eq!(map.insert(1, "ONE"), Some("one"));
        assert_eq!(map.get(&1), Some(&"ONE"));

        // 测试删除
        assert_eq!(map.remove(&1), Some("ONE"));
        assert_eq!(map.get(&1), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_rehash() {
        let mut map = HashMap::with_capacity(2);

        // 插入足够多的元素触发重哈希
        for i in 0..100 {
            map.insert(i, i.to_string());
        }

        // 验证所有元素都能正确访问
        for i in 0..100 {
            assert_eq!(map.get(&i), Some(&i.to_string()));
        }
    }

    #[test]
    fn test_iterators() {
        let mut map = HashMap::new();
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");

        // 测试不可变迭代器
        let mut pairs: Vec<_> = map.iter().collect();
        pairs.sort_by_key(|&(k, _)| *k);
        assert_eq!(pairs, vec![(&1, &"one"), (&2, &"two"), (&3, &"three")]);

        // 测试可变迭代器
        for (_, v) in map.iter_mut() {
            *v = "changed";
        }
        assert_eq!(map.get(&1), Some(&"changed"));
    }

    #[test]
    fn test_custom_hasher() {
        use std::collections::hash_map::DefaultHasher;

        let hash_builder = RandomState::new();
        let mut map = HashMap::with_capacity_and_hasher(10, hash_builder);

        map.insert("key1", 1);
        map.insert("key2", 2);

        assert_eq!(map.get("key1"), Some(&1));
        assert_eq!(map.get("key2"), Some(&2));
    }
}
