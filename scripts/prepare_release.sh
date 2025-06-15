#!/bin/bash

echo "🚀 准备发布 dstructs 到 crates.io"
echo "=================================="

# 1. 清理构建
echo "1. 清理构建..."
cargo clean

# 2. 代码格式化
echo "2. 格式化代码..."
cargo fmt --check || {
    echo "❌ 代码格式不正确，请运行 'cargo fmt'"
    exit 1
}

# 3. 代码检查
echo "3. 运行 clippy 检查..."
cargo clippy --all-targets --all-features -- -D warnings || {
    echo "❌ Clippy 检查失败"
    exit 1
}

# 4. 运行测试
echo "4. 运行测试..."
cargo test || {
    echo "❌ 测试失败"
    exit 1
}

# 5. 构建文档
echo "5. 构建文档..."
cargo doc --no-deps || {
    echo "❌ 文档构建失败"
    exit 1
}

# 6. 检查包内容
echo "6. 检查包内容..."
cargo package --list

# 7. 构建包
echo "7. 构建包..."
cargo package || {
    echo "❌ 包构建失败"
    exit 1
}

echo "✅ 所有检查通过！准备发布。"
echo ""
echo "下一步："
echo "1. 检查 target/package/dstructs-0.1.0.crate 包内容"
echo "2. 运行 'cargo publish --dry-run' 进行最终检查"
echo "3. 运行 'cargo publish' 发布到 crates.io" 