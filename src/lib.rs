//! # dstructs
//!
//! 一个高性能的 Rust 数据结构库，提供线程安全和高效的数据结构实现。
//!
//! ## 特性
//!
//! - 🚀 **高性能**: 优化的数据结构实现
//! - 🔒 **线程安全**: 支持并发操作的数据结构  
//! - 🛡️ **内存安全**: 100% 安全 Rust 代码，无 unsafe 块
//! - 📖 **完整文档**: 详细的 API 文档和使用示例
//!
//! ## 快速开始
//!
//! ```rust
//! use dstructs::prelude::*;
//!
//! fn main() {
//!     // 初始化库
//!     dstructs::init();
//!     
//!     // 使用数据结构
//!     // ... 你的代码
//! }
//! ```
//!
//! ## 模块
//!
//! - [`core`] - 核心数据结构实现
//! - [`prelude`] - 常用类型的重导出

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![deny(clippy::all)]

pub mod core;

/// 重导出常用类型
pub mod prelude {
    pub use crate::core::*;
}

/// 库的版本信息
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// 初始化库
///
/// 设置日志级别并初始化必要的组件
pub fn init() {
    env_logger::init();
}
