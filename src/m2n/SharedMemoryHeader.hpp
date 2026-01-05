#pragma once

#include <atomic>
#include <cstddef>

namespace precice::m2n {

/**
 * @brief 共享内存区域的头部元数据
 * * 用于协调代理进程与普通进程之间的数据交换。
 * 使用 alignas(64) 避免伪共享 (False Sharing) 问题，适配常见的 CPU 缓存行大小。
 */
struct alignas(64) SharedMemoryHeader {
    // 状态标志位
    // 0: 空闲 (Idle) - 可以写入
    // 1: 数据已就绪 (Data Ready) - 代理已写入，Worker 可读取
    // 2: 读取完成 (Read Done) - Worker 已读完，等待下一次写入
    std::atomic<int> stateFlag{0};

    // 有效载荷大小 (字节)
    std::atomic<size_t> payloadSize{0};

    // 辅助字段：例如记录当前的时间步或迭代步，防止逻辑错位
    std::atomic<int> timeStep{-1};
};

} // namespace precice::m2n