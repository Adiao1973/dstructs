# dstructs

一个高性能的 Rust 数据结构库，提供线程安全和高效的数据结构实现。

## 特性

- 🚀 **高性能**: 优化的数据结构实现
- 🔒 **线程安全**: 支持并发操作的数据结构
- 🛡️ **内存安全**: 100% 安全 Rust 代码，无 unsafe 块
- 📖 **完整文档**: 详细的 API 文档和使用示例

## 快速开始

在你的 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
dstructs = "0.1.0"
```

基本使用：

```rust
use dstructs::prelude::*;

fn main() {
    // 初始化库
    dstructs::init();
    
    // 使用数据结构
    // ... 你的代码
}
```

## 支持的数据结构

- 并发数据结构
- 高性能集合类型
- 专用算法实现

## 文档

详细文档请访问 [docs.rs/dstructs](https://docs.rs/dstructs)

## 许可证

本项目基于 MIT 许可证开源。详情请见 [LICENSE-MIT](LICENSE-MIT) 文件。

## 贡献

欢迎提交 Issues 和 Pull requests！

## 版本历史

- 0.1.0 - 初始版本 