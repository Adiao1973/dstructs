#![no_std]

extern crate alloc;

use crate::core::collections::{HashMap, Vector};
use core::cmp::Ordering as CmpOrdering;
use core::fmt::{self, Debug};
use core::hash::Hash;
use core::sync::atomic::{AtomicBool, Ordering};

/// 添加必要的导入
use alloc::collections::VecDeque;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::Add;
use hashbrown::HashSet;

/// 图的边结构
///
/// # 类型参数
/// * `T` - 节点类型
/// * `W` - 权重类型
#[derive(Clone)]
pub struct Edge<T, W> {
    /// 源节点
    pub source: T,
    /// 目标节点
    pub target: T,
    /// 边的权重
    pub weight: W,
}

impl<T: Debug, W: Debug> Debug for Edge<T, W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Edge")
            .field("source", &self.source)
            .field("target", &self.target)
            .field("weight", &self.weight)
            .finish()
    }
}

/// 图数据结构的实现
///
/// # 类型参数
/// * `T` - 节点类型，必须是可哈希、可克隆、可比较的
/// * `W` - 边的权重类型，必须是可克隆、可比较、可相加的
///
/// # 示例
/// ```
/// use quantum_data_structures::core::collections::Graph;
///
/// let mut graph = Graph::new(false); // 创建无向图
/// graph.add_vertex(1);
/// graph.add_vertex(2);
/// graph.add_edge(1, 2, 1.0);
/// ```
pub struct Graph<T, W>
where
    T: Hash + Eq + Clone,
    W: Clone,
{
    /// 邻接表表示
    adjacency_list: HashMap<T, Vector<Edge<T, W>>>,
    /// 是否是有向图
    is_directed: bool,
    /// 并发控制标志
    is_locked: AtomicBool,
}

impl<T, W> Debug for Graph<T, W>
where
    T: Hash + Eq + Clone + Debug,
    W: Clone + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Graph")
            .field("is_directed", &self.is_directed)
            .field("is_locked", &self.is_locked)
            .finish()
    }
}

impl<T, W> Graph<T, W>
where
    T: Hash + Eq + Clone,
    W: Clone + PartialOrd + Add<Output = W> + Copy,
{
    /// 创建新的图实例
    ///
    /// # 参数
    /// * `is_directed` - 是否是有向图
    pub fn new(is_directed: bool) -> Self {
        Self {
            adjacency_list: HashMap::new(),
            is_directed,
            is_locked: AtomicBool::new(false),
        }
    }

    /// 获取并发锁
    fn acquire_lock(&self) -> bool {
        !self.is_locked.swap(true, Ordering::Acquire)
    }

    /// 释放并发锁
    fn release_lock(&self) {
        self.is_locked.store(false, Ordering::Release);
    }

    /// 添加节点
    ///
    /// # 参数
    /// * `vertex` - 要添加的节点
    ///
    /// # 返回
    /// 如果节点已存在，返回 false；否则返回 true
    pub fn add_vertex(&mut self, vertex: T) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        let result = if self.adjacency_list.contains_key(&vertex) {
            false
        } else {
            self.adjacency_list.insert(vertex, Vector::new());
            true
        };

        self.release_lock();
        result
    }

    /// 添加边
    ///
    /// # 参数
    /// * `source` - 源节点
    /// * `target` - 目标节点
    /// * `weight` - 边的权重
    ///
    /// # 返回
    /// 如果添加成功返回 true，否则返回 false
    pub fn add_edge(&mut self, source: T, target: T, weight: W) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        let result = if !self.adjacency_list.contains_key(&source)
            || !self.adjacency_list.contains_key(&target)
        {
            false
        } else {
            let edge = Edge {
                source: source.clone(),
                target: target.clone(),
                weight: weight.clone(),
            };

            if let Some(edges) = self.adjacency_list.get_mut(&source) {
                edges.push(edge);
            }

            if !self.is_directed {
                let reverse_edge = Edge {
                    source: target,
                    target: source,
                    weight,
                };
                if let Some(edges) = self.adjacency_list.get_mut(&reverse_edge.source) {
                    edges.push(reverse_edge);
                }
            }
            true
        };

        self.release_lock();
        result
    }

    /// 获取节点的所有邻居
    ///
    /// # 参数
    /// * `vertex` - 目标节点
    ///
    /// # 返回
    /// 返回包含所有邻居节点及对应边的向量
    pub fn get_neighbors(&self, vertex: &T) -> Option<Vector<Edge<T, W>>> {
        if !self.acquire_lock() {
            return None;
        }

        let result = self.adjacency_list.get(vertex).map(|edges| edges.clone());
        self.release_lock();
        result
    }

    /// 删除节点
    ///
    /// # 参数
    /// * `vertex` - 要删除的节点
    ///
    /// # 返回
    /// 如果节点存在并被删除返回 true，否则返回 false
    pub fn remove_vertex(&mut self, vertex: &T) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        let result = if !self.adjacency_list.contains_key(vertex) {
            false
        } else {
            // 删除所有指向该节点的边
            let mut keys = Vector::new();
            for (k, _) in self.adjacency_list.iter() {
                keys.push(k.clone());
            }

            for key in keys.iter() {
                if let Some(edges) = self.adjacency_list.get_mut(key) {
                    let mut i = 0;
                    while i < edges.len() {
                        if &edges[i].target == vertex {
                            edges.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                }
            }

            // 删除节点及其所有出边
            self.adjacency_list.remove(vertex);
            true
        };

        self.release_lock();
        result
    }

    /// 删除边
    ///
    /// # 参数
    /// * `source` - 源节点
    /// * `target` - 目标节点
    ///
    /// # 返回
    /// 如果边存在并被删除返回 true，否则返回 false
    pub fn remove_edge(&mut self, source: &T, target: &T) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        let mut result = false;
        if let Some(edges) = self.adjacency_list.get_mut(source) {
            let initial_len = edges.len();
            let mut i = 0;
            while i < edges.len() {
                if &edges[i].target == target {
                    edges.remove(i);
                } else {
                    i += 1;
                }
            }
            result = initial_len != edges.len();

            if !self.is_directed && result {
                if let Some(target_edges) = self.adjacency_list.get_mut(target) {
                    let mut i = 0;
                    while i < target_edges.len() {
                        if &target_edges[i].target == source {
                            target_edges.remove(i);
                        } else {
                            i += 1;
                        }
                    }
                }
            }
        }

        self.release_lock();
        result
    }

    /// 获取图中的节点数量
    pub fn vertex_count(&self) -> usize {
        if !self.acquire_lock() {
            return 0;
        }
        let count = self.adjacency_list.len();
        self.release_lock();
        count
    }

    /// 获取图中的边数量
    pub fn edge_count(&self) -> usize {
        if !self.acquire_lock() {
            return 0;
        }
        let mut count = 0;
        for (_, edges) in self.adjacency_list.iter() {
            count += edges.len();
        }
        self.release_lock();
        if self.is_directed {
            count
        } else {
            count / 2
        }
    }

    /// 深度优先搜索遍历
    ///
    /// # 参数
    /// * `start` - 起始节点
    /// * `visitor` - 访问者函数，接收当前节点和其边的引用
    ///
    /// # 返回
    /// 返回访问过的节点列表
    pub fn dfs<F>(&self, start: &T, mut visitor: F) -> Option<Vec<T>>
    where
        F: FnMut(&T, &[Edge<T, W>]),
    {
        if !self.acquire_lock() {
            return None;
        }

        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut result = Vec::new();

        if !self.adjacency_list.contains_key(start) {
            self.release_lock();
            return None;
        }

        stack.push(start.clone());
        visited.insert(start.clone());

        while let Some(vertex) = stack.pop() {
            result.push(vertex.clone());

            if let Some(edges) = self.adjacency_list.get(&vertex) {
                visitor(&vertex, edges);

                for edge in edges.iter().rev() {
                    if !visited.contains(&edge.target) {
                        stack.push(edge.target.clone());
                        visited.insert(edge.target.clone());
                    }
                }
            }
        }

        self.release_lock();
        Some(result)
    }

    /// 广度优先搜索遍历
    ///
    /// # 参数
    /// * `start` - 起始节点
    /// * `visitor` - 访问者函数，接收当前节点和其边的引用
    ///
    /// # 返回
    /// 返回访问过的节点列表
    pub fn bfs<F>(&self, start: &T, mut visitor: F) -> Option<Vec<T>>
    where
        F: FnMut(&T, &[Edge<T, W>]),
    {
        if !self.acquire_lock() {
            return None;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        if !self.adjacency_list.contains_key(start) {
            self.release_lock();
            return None;
        }

        queue.push_back(start.clone());
        visited.insert(start.clone());

        while let Some(vertex) = queue.pop_front() {
            result.push(vertex.clone());

            if let Some(edges) = self.adjacency_list.get(&vertex) {
                visitor(&vertex, edges);

                for edge in edges.iter() {
                    if !visited.contains(&edge.target) {
                        queue.push_back(edge.target.clone());
                        visited.insert(edge.target.clone());
                    }
                }
            }
        }

        self.release_lock();
        Some(result)
    }

    /// 递归实现的深度优先搜索
    ///
    /// # 参数
    /// * `start` - 起始节点
    /// * `visitor` - 访问者函数，接收当前节点和其边的引用
    ///
    /// # 返回
    /// 返回访问过的节点列表
    pub fn dfs_recursive<F>(&self, start: &T, mut visitor: F) -> Option<Vec<T>>
    where
        F: FnMut(&T, &[Edge<T, W>]) + Clone,
    {
        if !self.acquire_lock() {
            return None;
        }

        let mut visited = HashSet::new();
        let mut result = Vec::new();

        if !self.adjacency_list.contains_key(start) {
            self.release_lock();
            return None;
        }

        self.dfs_recursive_helper(start, &mut visitor, &mut visited, &mut result);
        self.release_lock();
        Some(result)
    }

    /// 递归 DFS 的辅助函数
    fn dfs_recursive_helper<F>(
        &self,
        vertex: &T,
        visitor: &mut F,
        visited: &mut HashSet<T>,
        result: &mut Vec<T>,
    ) where
        F: FnMut(&T, &[Edge<T, W>]) + Clone,
    {
        visited.insert(vertex.clone());
        result.push(vertex.clone());

        if let Some(edges) = self.adjacency_list.get(vertex) {
            visitor(vertex, edges);

            for edge in edges.iter() {
                if !visited.contains(&edge.target) {
                    self.dfs_recursive_helper(&edge.target, visitor, visited, result);
                }
            }
        }
    }

    /// 使用 Dijkstra 算法计算最短路径
    ///
    /// # 参数
    /// * `start` - 起始节点
    /// * `end` - 目标节点
    /// * `zero` - 权重的零值（用作起始距离）
    ///
    /// # 返回
    /// 返回一个元组 `(Option<Vec<T>>, Option<W>)`，其中：
    /// - 第一个元素是最短路径的节点序列（如果存在）
    /// - 第二个元素是最短路径的总权重（如果存在）
    pub fn shortest_path(&self, start: &T, end: &T, zero: W) -> (Option<Vec<T>>, Option<W>) {
        if !self.acquire_lock() {
            return (None, None);
        }

        // 检查起点和终点是否存在
        if !self.adjacency_list.contains_key(start) || !self.adjacency_list.contains_key(end) {
            self.release_lock();
            return (None, None);
        }

        // 初始化距离表和前驱节点表
        let mut distances: HashMap<T, W> = HashMap::new();
        let mut predecessors: HashMap<T, T> = HashMap::new();
        let mut unvisited = HashSet::new();

        // 初始化所有节点
        for (vertex, _) in self.adjacency_list.iter() {
            unvisited.insert(vertex.clone());
        }

        // 设置起点距离为 0
        distances.insert(start.clone(), zero);

        // Dijkstra 算法主循环
        while !unvisited.is_empty() {
            // 在未访问的节点中找到距离最小的
            let current = unvisited
                .iter()
                .min_by(|a, b| {
                    let dist_a = distances.get(*a);
                    let dist_b = distances.get(*b);
                    match (dist_a, dist_b) {
                        (Some(da), Some(db)) => da.partial_cmp(db).unwrap_or(CmpOrdering::Equal),
                        (Some(_), None) => CmpOrdering::Less,
                        (None, Some(_)) => CmpOrdering::Greater,
                        (None, None) => CmpOrdering::Equal,
                    }
                })
                .map(|x| x.clone());

            if let Some(current_vertex) = current {
                // 如果当前节点的距离是无穷大，说明无法到达更多节点
                if !distances.contains_key(&current_vertex) {
                    self.release_lock();
                    return (None, None);
                }

                // 如果找到了目标节点，构建路径并返回
                if &current_vertex == end {
                    let mut path = Vec::new();
                    let mut current = end.clone();
                    path.push(current.clone());

                    while let Some(predecessor) = predecessors.get(&current) {
                        path.push(predecessor.clone());
                        current = predecessor.clone();
                        if &current == start {
                            break;
                        }
                    }

                    path.reverse();
                    let total_distance = distances.get(end).copied();
                    self.release_lock();
                    return (Some(path), total_distance);
                }

                unvisited.remove(&current_vertex);

                // 更新邻居节点的距离
                if let Some(edges) = self.adjacency_list.get(&current_vertex) {
                    let current_distance = *distances.get(&current_vertex).unwrap();
                    for edge in edges {
                        let new_distance = current_distance + edge.weight;

                        match distances.get(&edge.target) {
                            None => {
                                distances.insert(edge.target.clone(), new_distance);
                                predecessors.insert(edge.target.clone(), current_vertex.clone());
                            }
                            Some(&current_distance) => {
                                if new_distance < current_distance {
                                    distances.insert(edge.target.clone(), new_distance);
                                    predecessors
                                        .insert(edge.target.clone(), current_vertex.clone());
                                }
                            }
                        }
                    }
                }
            } else {
                break;
            }
        }

        self.release_lock();
        (None, None) // 没有找到路径
    }

    /// 检查图是否连通
    ///
    /// 对于无向图，检查是否所有节点都可以互相到达
    /// 对于有向图，检查是否存在一个节点可以到达所有其他节点
    ///
    /// # 返回
    /// 如果图是连通的返回 true，否则返回 false
    pub fn is_connected(&self) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        if self.adjacency_list.is_empty() {
            self.release_lock();
            return true;
        }

        // 从第一个节点开始进行 BFS
        let start = self.adjacency_list.iter().next().unwrap().0.clone();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start.clone());
        visited.insert(start);

        while let Some(vertex) = queue.pop_front() {
            if let Some(edges) = self.adjacency_list.get(&vertex) {
                for edge in edges {
                    if !visited.contains(&edge.target) {
                        visited.insert(edge.target.clone());
                        queue.push_back(edge.target.clone());
                    }
                }
            }
        }

        // 检查是否所有节点都被访问到
        let is_connected = visited.len() == self.adjacency_list.len();
        self.release_lock();
        is_connected
    }

    /// 检查有向图是否强连通
    ///
    /// 一个有向图是强连通的，当且仅当从任意节点出发都能到达其他所有节点
    ///
    /// # 返回
    /// 如果图是强连通的返回 true，否则返回 false
    pub fn is_strongly_connected(&self) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        if !self.is_directed {
            self.release_lock();
            return self.is_connected();
        }

        if self.adjacency_list.is_empty() {
            self.release_lock();
            return true;
        }

        // 对每个节点进行 BFS，检查是否能到达所有其他节点
        for (start, _) in self.adjacency_list.iter() {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();

            queue.push_back(start.clone());
            visited.insert(start.clone());

            while let Some(vertex) = queue.pop_front() {
                if let Some(edges) = self.adjacency_list.get(&vertex) {
                    for edge in edges {
                        if !visited.contains(&edge.target) {
                            visited.insert(edge.target.clone());
                            queue.push_back(edge.target.clone());
                        }
                    }
                }
            }

            // 如果从当前节点不能到达所有其他节点，则不是强连通的
            if visited.len() != self.adjacency_list.len() {
                self.release_lock();
                return false;
            }
        }

        self.release_lock();
        true
    }

    /// 检测图中是否存在环
    ///
    /// 对于无向图，使用 DFS 并记录父节点来检测环
    /// 对于有向图，调用 has_directed_cycle 方法
    ///
    /// # 返回
    /// 如果存在环返回 true，否则返回 false
    pub fn has_cycle(&self) -> bool {
        if !self.acquire_lock() {
            return false;
        }

        if self.is_directed {
            let result = self.has_directed_cycle();
            self.release_lock();
            return result;
        }

        if self.adjacency_list.is_empty() {
            self.release_lock();
            return false;
        }

        let mut visited = HashSet::new();
        let mut parent: HashMap<T, T> = HashMap::new();

        // 对每个未访问的节点进行 DFS
        for start in self.adjacency_list.iter().map(|(k, _)| k) {
            if !visited.contains(start) {
                if self.has_cycle_dfs(start, &mut visited, &mut parent) {
                    self.release_lock();
                    return true;
                }
            }
        }

        self.release_lock();
        false
    }

    /// 检测无向图中是否存在环的 DFS 辅助函数
    fn has_cycle_dfs(
        &self,
        vertex: &T,
        visited: &mut HashSet<T>,
        parent: &mut HashMap<T, T>,
    ) -> bool {
        visited.insert(vertex.clone());

        if let Some(edges) = self.adjacency_list.get(vertex) {
            for edge in edges {
                let neighbor = &edge.target;

                // 检查自环
                if vertex == neighbor {
                    return true;
                }

                if !visited.contains(neighbor) {
                    parent.insert(neighbor.clone(), vertex.clone());
                    if self.has_cycle_dfs(neighbor, visited, parent) {
                        return true;
                    }
                } else if let Some(p) = parent.get(vertex) {
                    // 如果邻居节点已访问且不是当前节点的父节点，说明存在环
                    if neighbor != p {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// 检测有向图中是否存在环
    ///
    /// 使用 DFS 并维护三种状态的访问标记：
    /// - 未访问
    /// - 正在访问（在当前 DFS 路径上）
    /// - 已完成访问
    ///
    /// # 返回
    /// 如果存在环返回 true，否则返回 false
    pub fn has_directed_cycle(&self) -> bool {
        if !self.is_directed {
            return self.has_cycle();
        }

        if self.adjacency_list.is_empty() {
            return false;
        }

        let mut visited = HashSet::new(); // 已完成访问的节点
        let mut on_path = HashSet::new(); // 当前 DFS 路径上的节点

        // 对每个未访问的节点进行 DFS
        for start in self.adjacency_list.iter().map(|(k, _)| k) {
            if !visited.contains(start) && !on_path.contains(start) {
                if self.has_directed_cycle_dfs(start, &mut visited, &mut on_path) {
                    return true;
                }
            }
        }

        false
    }

    /// 检测有向图中是否存在环的 DFS 辅助函数
    fn has_directed_cycle_dfs(
        &self,
        vertex: &T,
        visited: &mut HashSet<T>,
        on_path: &mut HashSet<T>,
    ) -> bool {
        on_path.insert(vertex.clone());

        if let Some(edges) = self.adjacency_list.get(vertex) {
            for edge in edges {
                let neighbor = &edge.target;

                if on_path.contains(neighbor) {
                    // 如果邻居节点在当前路径上，说明存在环
                    return true;
                }

                if !visited.contains(neighbor) && !on_path.contains(neighbor) {
                    if self.has_directed_cycle_dfs(neighbor, visited, on_path) {
                        return true;
                    }
                }
            }
        }

        on_path.remove(vertex);
        visited.insert(vertex.clone());
        false
    }

    /// 计算图的拓扑排序
    ///
    /// 使用 Kahn 算法实现拓扑排序。只适用于有向无环图(DAG)。
    /// 如果图中存在环，返回 None。
    ///
    /// # 返回值
    /// - Some(Vec<T>) - 一个合法的拓扑序列
    /// - None - 如果图不是 DAG（存在环）
    pub fn topological_sort(&self) -> Option<Vec<T>> {
        if !self.acquire_lock() {
            return None;
        }

        // 如果不是有向图，返回 None
        if !self.is_directed {
            self.release_lock();
            return None;
        }

        // 计算每个节点的入度
        let mut in_degree: HashMap<T, usize> = HashMap::new();
        for vertex in self.adjacency_list.iter().map(|(k, _)| k) {
            in_degree.insert(vertex.clone(), 0);
        }

        // 统计入度
        for (_, edges) in self.adjacency_list.iter() {
            for edge in edges {
                if let Some(count) = in_degree.get_mut(&edge.target) {
                    *count += 1;
                }
            }
        }

        // 将所有入度为 0 的节点加入队列
        let mut queue = VecDeque::new();
        for (vertex, &degree) in in_degree.iter() {
            if degree == 0 {
                queue.push_back(vertex.clone());
            }
        }

        let mut result = Vec::new();

        // 主循环
        while let Some(vertex) = queue.pop_front() {
            result.push(vertex.clone());

            // 减少所有相邻节点的入度
            if let Some(edges) = self.adjacency_list.get(&vertex) {
                for edge in edges {
                    if let Some(count) = in_degree.get_mut(&edge.target) {
                        *count -= 1;
                        // 如果入度变为 0，加入队列
                        if *count == 0 {
                            queue.push_back(edge.target.clone());
                        }
                    }
                }
            }
        }

        // 如果不是所有节点都被访问到，说明图中有环
        let success = result.len() == self.adjacency_list.len();
        self.release_lock();

        if success {
            Some(result)
        } else {
            None
        }
    }
}

impl<T: Hash + Eq + Clone, W: Clone> Clone for Graph<T, W>
where
    T: Hash + Eq + Clone,
    W: Clone,
{
    fn clone(&self) -> Self {
        Self {
            adjacency_list: self.adjacency_list.clone(),
            is_directed: self.is_directed,
            is_locked: AtomicBool::new(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;
    use core::cell::RefCell;

    #[test]
    fn test_graph_creation() {
        let mut graph: Graph<i32, f64> = Graph::new(false);
        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_vertex() {
        let mut graph: Graph<i32, f64> = Graph::new(false);
        assert!(graph.add_vertex(1));
        assert!(!graph.add_vertex(1));
        assert_eq!(graph.vertex_count(), 1);
    }

    #[test]
    fn test_add_edge() {
        let mut graph: Graph<i32, f64> = Graph::new(false);
        graph.add_vertex(1);
        graph.add_vertex(2);
        assert!(graph.add_edge(1, 2, 1.0));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_remove_vertex() {
        let mut graph: Graph<i32, f64> = Graph::new(false);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(1, 2, 1.0);
        assert!(graph.remove_vertex(&1));
        assert_eq!(graph.vertex_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_remove_edge() {
        let mut graph: Graph<i32, f64> = Graph::new(false);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(1, 2, 1.0);
        assert!(graph.remove_edge(&1, &2));
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_dfs() {
        let mut graph = Graph::new(true);

        // 构建测试图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 4, 1.0);
        graph.add_edge(3, 4, 1.0);

        let mut visited_order = Vec::new();
        let result = graph.dfs(&1, |&node, _| {
            visited_order.push(node);
        });

        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(visited_order.len(), 4);
        assert_eq!(visited_order[0], 1);
        assert!(visited_order.contains(&2));
        assert!(visited_order.contains(&3));
        assert!(visited_order.contains(&4));
    }

    #[test]
    fn test_bfs() {
        let mut graph = Graph::new(true);

        // 构建测试图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 4, 1.0);
        graph.add_edge(3, 4, 1.0);

        let mut visited_order = Vec::new();
        let result = graph.bfs(&1, |&node, _| {
            visited_order.push(node);
        });

        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.len(), 4);
        assert_eq!(visited_order.len(), 4);
        assert_eq!(visited_order[0], 1);
        assert!(visited_order[1] == 2 || visited_order[1] == 3);
        assert!(visited_order[2] == 2 || visited_order[2] == 3);
        assert_eq!(visited_order[3], 4);
    }

    #[test]
    fn test_dfs_recursive() {
        let mut graph = Graph::new(true);

        // 构建测试图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 4, 1.0);
        graph.add_edge(3, 4, 1.0);

        let visited_order = RefCell::new(Vec::new());
        let result = graph.dfs_recursive(&1, |node, _| {
            visited_order.borrow_mut().push(*node);
        });

        assert!(result.is_some());
        let path = result.unwrap();
        assert_eq!(path.len(), 4);

        let visited = visited_order.into_inner();
        assert_eq!(visited.len(), 4);
        assert_eq!(visited[0], 1);
        assert!(visited.contains(&2));
        assert!(visited.contains(&3));
        assert!(visited.contains(&4));
    }

    #[test]
    fn test_graph_traversal_with_cycles() {
        let mut graph = Graph::new(true);

        // 构建带环的测试图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 1, 1.0); // 创建环

        // 测试 DFS
        let dfs_result = graph.dfs(&1, |_, _| {});
        assert!(dfs_result.is_some());
        let dfs_path = dfs_result.unwrap();
        assert_eq!(dfs_path.len(), 3);

        // 测试 BFS
        let bfs_result = graph.bfs(&1, |_, _| {});
        assert!(bfs_result.is_some());
        let bfs_path = bfs_result.unwrap();
        assert_eq!(bfs_path.len(), 3);
    }

    #[test]
    fn test_disconnected_graph() {
        let mut graph = Graph::new(true);

        // 创建非连通图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(3, 4, 1.0); // 独立的边

        // 测试 DFS
        let dfs_result = graph.dfs(&1, |_, _| {});
        assert!(dfs_result.is_some());
        let dfs_path = dfs_result.unwrap();
        assert_eq!(dfs_path.len(), 2); // 只能访问到 1 和 2

        // 测试 BFS
        let bfs_result = graph.bfs(&1, |_, _| {});
        assert!(bfs_result.is_some());
        let bfs_path = bfs_result.unwrap();
        assert_eq!(bfs_path.len(), 2); // 只能访问到 1 和 2
    }

    #[test]
    fn test_shortest_path() {
        let mut graph = Graph::new(true);

        // 构建测试图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 4.0);
        graph.add_edge(1, 3, 2.0);
        graph.add_edge(3, 2, 1.0);
        graph.add_edge(2, 4, 3.0);
        graph.add_edge(3, 4, 5.0);

        // 测试最短路径：1 -> 3 -> 2 -> 4
        let (path, distance) = graph.shortest_path(&1, &4, 0.0);
        assert!(path.is_some());
        assert!(distance.is_some());

        let path = path.unwrap();
        let distance = distance.unwrap();

        assert_eq!(path, vec![1, 3, 2, 4]);
        assert_eq!(distance, 6.0); // 2.0 + 1.0 + 3.0 = 6.0
    }

    #[test]
    fn test_shortest_path_no_path() {
        let mut graph = Graph::new(true);

        // 构建不连通的图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);

        graph.add_edge(1, 2, 1.0);
        // 3 没有连接到任何节点

        let (path, distance) = graph.shortest_path(&1, &3, 0.0);
        assert!(path.is_none());
        assert!(distance.is_none());
    }

    #[test]
    fn test_shortest_path_with_cycles() {
        let mut graph = Graph::new(true);

        // 构建带环的图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 2.0);
        graph.add_edge(3, 1, 4.0); // 创建环
        graph.add_edge(1, 3, 5.0); // 直接路径

        let (path, distance) = graph.shortest_path(&1, &3, 0.0);
        assert!(path.is_some());
        assert!(distance.is_some());

        let path = path.unwrap();
        let distance = distance.unwrap();

        assert_eq!(path, vec![1, 2, 3]); // 应该选择 1->2->3 而不是 1->3
        assert_eq!(distance, 3.0); // 1.0 + 2.0 = 3.0
    }

    #[test]
    fn test_is_connected() {
        let mut graph = Graph::new(false);

        // 空图被认为是连通的
        assert!(graph.is_connected());

        // 单个节点是连通的
        graph.add_vertex(1);
        assert!(graph.is_connected());

        // 添加连通的节点
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        assert!(graph.is_connected());

        // 添加不连通的节点
        graph.add_vertex(4);
        assert!(!graph.is_connected());

        // 连接后变为连通
        graph.add_edge(3, 4, 1.0);
        assert!(graph.is_connected());
    }

    #[test]
    fn test_is_strongly_connected() {
        let mut graph = Graph::new(true);

        // 空图被认为是强连通的
        assert!(graph.is_strongly_connected());

        // 单个节点是强连通的
        graph.add_vertex(1);
        assert!(graph.is_strongly_connected());

        // 添加节点和单向边，不是强连通的
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        assert!(!graph.is_strongly_connected());

        // 添加返回边，成为强连通图
        graph.add_edge(3, 1, 1.0);
        assert!(graph.is_strongly_connected());

        // 添加不连通的节点
        graph.add_vertex(4);
        assert!(!graph.is_strongly_connected());

        // 添加双向连接，恢复强连通性
        graph.add_edge(3, 4, 1.0);
        graph.add_edge(4, 1, 1.0);
        assert!(graph.is_strongly_connected());
    }

    #[test]
    fn test_has_cycle_undirected() {
        let mut graph = Graph::new(false);

        // 空图没有环
        assert!(!graph.has_cycle());

        // 单个节点没有环
        graph.add_vertex(1);
        assert!(!graph.has_cycle());

        // 两个连通节点没有环
        graph.add_vertex(2);
        graph.add_edge(1, 2, 1.0);
        assert!(!graph.has_cycle());

        // 添加边形成环
        graph.add_vertex(3);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 1, 1.0);
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_has_directed_cycle() {
        let mut graph = Graph::new(true);

        // 空图没有环
        assert!(!graph.has_directed_cycle());

        // 单个节点没有环
        graph.add_vertex(1);
        assert!(!graph.has_directed_cycle());

        // 两个节点的有向边没有环
        graph.add_vertex(2);
        graph.add_edge(1, 2, 1.0);
        assert!(!graph.has_directed_cycle());

        // 添加返回边形成环
        graph.add_edge(2, 1, 1.0);
        assert!(graph.has_directed_cycle());

        // 测试更复杂的环
        graph.add_vertex(3);
        graph.add_vertex(4);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 4, 1.0);
        graph.add_edge(4, 2, 1.0);
        assert!(graph.has_directed_cycle());
    }

    #[test]
    fn test_cycle_with_self_loop() {
        let mut graph = Graph::new(true);

        graph.add_vertex(1);
        graph.add_edge(1, 1, 1.0); // 自环
        assert!(graph.has_directed_cycle());

        let mut undirected = Graph::new(false);
        undirected.add_vertex(1);
        undirected.add_edge(1, 1, 1.0); // 自环
        assert!(undirected.has_cycle());
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = Graph::new(true);

        // 构建一个 DAG
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_vertex(4);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 4, 1.0);
        graph.add_edge(3, 4, 1.0);

        let result = graph.topological_sort();
        assert!(result.is_some());
        let order = result.unwrap();

        // 验证拓扑序列的正确性
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], 1); // 1 必须在最前面
        assert_eq!(order[3], 4); // 4 必须在最后面
                                 // 2 和 3 的顺序可以互换
        assert!((order[1] == 2 && order[2] == 3) || (order[1] == 3 && order[2] == 2));
    }

    #[test]
    fn test_topological_sort_with_cycle() {
        let mut graph = Graph::new(true);

        // 构建一个有环的图
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 1, 1.0); // 形成环

        let result = graph.topological_sort();
        assert!(result.is_none()); // 有环图应该返回 None
    }

    #[test]
    fn test_topological_sort_empty_and_single() {
        let mut graph = Graph::<i32, i32>::new(true);

        // 空图
        let result = graph.topological_sort();
        assert_eq!(result.unwrap(), vec![]);

        // 单个节点
        graph.add_vertex(1);
        let result = graph.topological_sort();
        assert_eq!(result.unwrap(), vec![1]);
    }

    #[test]
    fn test_topological_sort_undirected() {
        let mut graph = Graph::new(false);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(1, 2, 1.0);

        let result = graph.topological_sort();
        assert!(result.is_none()); // 无向图应该返回 None
    }
}
