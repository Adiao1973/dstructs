//! 并发数据结构模块
//!
//! 本模块提供了一系列无锁并发数据结构的实现，包括：
//! - 无锁队列 (LockFreeQueue)
//! - 无锁栈 (计划中)
//! - 并发哈希表 (计划中)
//!
//! 所有实现都保证：
//! - 线程安全
//! - 内存安全
//! - 无数据竞争
//! - 高性能

/// 无锁队列的实现
pub mod lock_free_queue;

pub use lock_free_queue::LockFreeQueue;

mod concurrent_hash_map;
pub use concurrent_hash_map::ConcurrentHashMap;
