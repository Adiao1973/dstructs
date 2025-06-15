use dashmap::DashMap;
use parking_lot::RwLock as PLRwLock;
use rand::Rng;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// 分段并发哈希表
#[derive(Debug)]
pub struct ConcurrentHashMap<K, V, S = RandomState>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    S: BuildHasher + Clone + Send + Sync + 'static,
{
    segments: Vec<Arc<Segment<K, V, S>>>,
    hasher_builder: S,
    concurrency_level: usize,
}

/// 存储段实现
#[derive(Debug)]
struct Segment<K, V, S>
where
    K: Hash + Eq + Send + Sync + 'static,
    V: Send + Sync + 'static,
    S: BuildHasher + Clone + Send + Sync + 'static,
{
    data: DashMap<K, V, S>,
    stats: PLRwLock<SegmentStats>,
}

/// 段统计信息
#[derive(Debug, Default, Clone)]
struct SegmentStats {
    operations: u64,
    contention_count: u64,
    last_resize: Option<Instant>,
    avg_operation_time: Duration,
}

impl<K, V, S> ConcurrentHashMap<K, V, S>
where
    K: Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
    S: BuildHasher + Clone + Send + Sync + Default + 'static,
{
    /// 创建新的并发哈希表
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(16, S::default())
    }

    /// 使用指定容量和哈希器创建哈希表
    pub fn with_capacity_and_hasher(capacity: usize, hasher_builder: S) -> Self {
        let concurrency_level = num_cpus::get().max(16);
        let segment_capacity = (capacity + concurrency_level - 1) / concurrency_level;

        let segments = (0..concurrency_level)
            .map(|_| {
                Arc::new(Segment {
                    data: DashMap::with_capacity_and_hasher(
                        segment_capacity,
                        hasher_builder.clone(),
                    ),
                    stats: PLRwLock::new(SegmentStats::default()),
                })
            })
            .collect();

        Self {
            segments,
            hasher_builder,
            concurrency_level,
        }
    }

    /// 插入键值对
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let start = Instant::now();
        let segment_index = self.get_segment_index(&key);
        let segment = &self.segments[segment_index];

        let result = segment.data.insert(key, value);

        let mut stats = segment.stats.write();
        stats.operations += 1;
        stats.avg_operation_time = (stats.avg_operation_time + start.elapsed()) / 2;

        result
    }

    /// 获取值
    pub fn get(&self, key: &K) -> Option<V> {
        let start = Instant::now();
        let segment_index = self.get_segment_index(key);
        let segment = &self.segments[segment_index];

        let result = segment.data.get(key).map(|v| v.clone());

        let mut stats = segment.stats.write();
        stats.operations += 1;
        stats.avg_operation_time = (stats.avg_operation_time + start.elapsed()) / 2;

        result
    }

    /// 删除键值对
    pub fn remove(&self, key: &K) -> Option<V> {
        let start = Instant::now();
        let segment_index = self.get_segment_index(key);
        let segment = &self.segments[segment_index];

        let result = segment.data.remove(key).map(|(_, v)| v);

        let mut stats = segment.stats.write();
        stats.operations += 1;
        stats.avg_operation_time = (stats.avg_operation_time + start.elapsed()) / 2;

        result
    }

    /// 获取性能统计信息
    pub fn get_stats(&self) -> HashMap<usize, SegmentStats> {
        self.segments
            .iter()
            .enumerate()
            .map(|(i, segment)| (i, segment.stats.read().clone()))
            .collect::<HashMap<_, _>>()
    }

    // 私有辅助方法
    fn get_segment_index(&self, key: &K) -> usize {
        let mut hasher = self.hasher_builder.build_hasher();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.concurrency_level
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_operations() {
        let map = ConcurrentHashMap::<i32, String>::new();

        // 测试插入
        assert_eq!(map.insert(1, "one".to_string()), None);
        assert_eq!(map.insert(2, "two".to_string()), None);

        // 测试获取
        assert_eq!(map.get(&1), Some("one".to_string()));
        assert_eq!(map.get(&2), Some("two".to_string()));
        assert_eq!(map.get(&3), None);

        // 测试更新
        assert_eq!(map.insert(1, "ONE".to_string()), Some("one".to_string()));
        assert_eq!(map.get(&1), Some("ONE".to_string()));

        // 测试删除
        assert_eq!(map.remove(&1), Some("ONE".to_string()));
        assert_eq!(map.get(&1), None);
    }

    #[test]
    fn test_concurrent_operations() {
        let map = Arc::new(ConcurrentHashMap::<i32, i32>::new());
        let threads = 8;
        let operations = 5000;
        let mut handles = vec![];

        // 写入线程
        for t in 0..threads {
            let map = Arc::clone(&map);
            let handle = thread::spawn(move || {
                for i in 0..operations {
                    let key = i * threads + t;
                    map.insert(key as i32, key as i32);
                }
            });
            handles.push(handle);
        }

        // 读取线程 - 优化版本
        for _ in 0..threads {
            let map = Arc::clone(&map);
            let handle = thread::spawn(move || {
                let mut rng = rand::thread_rng();
                for _ in 0..operations {
                    // 随机读取一个范围内的key，避免遍历所有key
                    let key = rng.gen_range(0..(operations * threads)) as i32;
                    let _ = map.get(&key);
                }
            });
            handles.push(handle);
        }

        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }

        // 验证写入的值
        for t in 0..threads {
            for i in 0..operations {
                let key = i * threads + t;
                assert_eq!(map.get(&(key as i32)), Some(key as i32));
            }
        }
    }

    // 添加一个更全面的性能测试
    #[test]
    fn test_concurrent_operations_with_metrics() {
        let map = Arc::new(ConcurrentHashMap::<i32, i32>::new());
        let threads = 8;
        let operations = 5000;

        let start = Instant::now();
        let write_stats = Arc::new(PLRwLock::new(Vec::new()));
        let read_stats = Arc::new(PLRwLock::new(Vec::new()));
        let mut handles = vec![];

        // 写入线程
        for t in 0..threads {
            let map = Arc::clone(&map);
            let write_stats = Arc::clone(&write_stats);
            let handle = thread::spawn(move || {
                let thread_start = Instant::now();
                for i in 0..operations {
                    let key = i * threads + t;
                    map.insert(key as i32, key as i32);
                }
                write_stats.write().push(thread_start.elapsed());
            });
            handles.push(handle);
        }

        // 读取线程
        for _ in 0..threads {
            let map = Arc::clone(&map);
            let read_stats = Arc::clone(&read_stats);
            let handle = thread::spawn(move || {
                let thread_start = Instant::now();
                let mut rng = rand::thread_rng();
                for _ in 0..operations {
                    let key = rng.gen_range(0..(operations * threads)) as i32;
                    let _ = map.get(&key);
                }
                read_stats.write().push(thread_start.elapsed());
            });
            handles.push(handle);
        }

        // 等待所有线程完成
        for handle in handles {
            handle.join().unwrap();
        }

        let total_time = start.elapsed();

        // 输出详细的性能指标
        println!("\nPerformance Metrics:");
        println!("Total time: {:?}", total_time);
        println!(
            "Operations per second: {:.2}",
            (threads * operations * 2) as f64 / total_time.as_secs_f64()
        );

        println!("\nWrite Thread Statistics:");
        let write_times = write_stats.read();
        println!(
            "Average write thread time: {:?}",
            write_times.iter().sum::<Duration>() / write_times.len() as u32
        );

        println!("\nRead Thread Statistics:");
        let read_times = read_stats.read();
        println!(
            "Average read thread time: {:?}",
            read_times.iter().sum::<Duration>() / read_times.len() as u32
        );

        // 验证数据一致性
        let mut errors = 0;
        for t in 0..threads {
            for i in 0..operations {
                let key = i * threads + t;
                if map.get(&(key as i32)) != Some(key as i32) {
                    errors += 1;
                }
            }
        }
        assert_eq!(errors, 0, "发现 {} 个数据不一致", errors);
    }
}
