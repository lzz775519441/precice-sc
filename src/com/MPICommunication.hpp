#ifndef PRECICE_NO_MPI

#pragma once

#include <mpi.h>
#include <string>

#include "com/Communication.hpp"
#include "com/SharedPointer.hpp"
#include "logging/Logger.hpp"
#include "precice/impl/Types.hpp"

namespace precice::com {
/**
 * @brief Provides implementation for basic MPI point-to-point communication.
 *
 * The methods for establishing a connection between two coupling participants
 * are not implemented and left to subclasses.
 */
class MPICommunication : public ::precice::com::Communication {
public:
  MPICommunication();

  /// Destructor, empty.
  ~MPICommunication() override;

  /**
   * @brief Sends a std::string to process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void send(std::string const &itemToSend, Rank rankReceiver) override;

  /// Sends an array of integer values.
  void send(precice::span<const int> itemsToSend, Rank rankReceiver) override;

  /// Asynchronously sends an array of integer values.
  PtrRequest aSend(precice::span<const int> itemsToSend, Rank rankReceiver) override;

  /// Sends an array of double values.
  void send(precice::span<const double> itemsToSend, Rank rankReceiver) override;

  /// Asynchronously sends an array of double values.
  PtrRequest aSend(precice::span<const double> itemsToSend, Rank rankReceiver) override;

  /**
   * @brief Sends a double to process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void send(double itemToSend, Rank rankReceiver) override;

  /// Asynchronously sends a double to process with given rank.
  PtrRequest aSend(const double &itemToSend, Rank rankReceiver) override;

  /**
   * @brief Sends an int to process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void send(int itemToSend, Rank rankReceiver) override;

  /// Asynchronously sends an int to process with given rank.
  PtrRequest aSend(const int &itemToSend, Rank rankReceiver) override;

  /**
   * @brief Sends a bool to process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void send(bool itemToSend, Rank rankReceiver) override;

  /// Asynchronously sends a bool to process with given rank.
  PtrRequest aSend(const bool &itemToSend, Rank rankReceiver) override;

  /**
   * @brief Receives a std::string from process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void receive(std::string &itemToReceive, Rank rankSender) override;

  /// Receives an array of integer values.
  void receive(precice::span<int> itemsToReceive, Rank rankSender) override;

  /// Receives an array of double values.
  void receive(precice::span<double> itemsToReceive, Rank rankSender) override;

  /// Asynchronously receives an array of double values.
  PtrRequest aReceive(precice::span<double> itemsToReceive, int rankSender) override;

  /**
   * @brief Receives a double from process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void receive(double &itemToReceive, Rank rankSender) override;

  /// Asynchronously receives a double from process with given rank.
  PtrRequest aReceive(double &itemToReceive, Rank rankSender) override;

  /**
   * @brief Receives an int from process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void receive(int &itemToReceive, Rank rankSender) override;

  /// Asynchronously receives an int from process with given rank.
  PtrRequest aReceive(int &itemToReceive, Rank rankSender) override;

  /**
   * @brief Receives a bool from process with given rank.
   *
   * Default MPI point-to-point communication is used.
   */
  void receive(bool &itemToReceive, Rank rankSender) override;

  /// Asynchronously receives a bool from process with given rank.
  PtrRequest aReceive(bool &itemToReceive, Rank rankSender) override;

  // ----------------------------------------------------------------
  // [新增代码] 拓扑感知相关接口
  // ----------------------------------------------------------------

  /**
   * @brief 基于物理节点分裂通信器
   * * 利用 MPI_Comm_split_type 将全局通信器分裂为多个节点内通信器。
   * 同一个物理节点上的进程将位于同一个 _localComm 中。
   *
   * @param globalComm 原始的父通信器（通常包含所有相关进程）
   */
  void splitCommunicatorByNode(MPI_Comm globalComm);

  /**
   * @brief 获取当前进程在本地节点上的 Rank (Local Rank)
   * * @return int 0 到 localSize-1
   */
  int getLocalRank() const;

  /**
   * @brief 获取本地节点上的进程总数
   */
  int getLocalSize() const;

  /**
   * @brief 判断当前进程是否为本地节点的“代理进程” (Proxy)
   * * 通常我们将 Local Rank 0 选为代理，负责跨节点通信或管理共享内存。
   */
  bool isLocalRank0() const;

  /**
   * @brief 获取节点内通信器句柄
   */
  MPI_Comm getLocalCommunicator() const;

  // ----------------------------------------------------------------
  // [新增代码] 共享内存管理接口 (第二步)
  // ----------------------------------------------------------------

  /**
   * @brief 在节点内分配共享内存窗口
   * * @param bytes 要分配的字节数。
   * 注意：通常只有代理进程 (LocalRank 0) 需要传入实际大小（作为缓冲区），
   * 普通进程传入 0 即可（它们不提供内存，只访问代理的内存）。
   */
  void allocateSharedMemoryWindow(MPI_Aint bytes);

  /**
   * @brief 获取指定本地进程的共享内存起始指针
   * * 利用 MPI_Win_shared_query 获取同节点其他进程的内存地址。
   * * @param targetLocalRank 目标进程在 _localComm 中的 Rank (默认 0，即获取代理进程的地址)
   * @return void* 指向共享内存的裸指针
   */
  void* getSharedMemoryPointer(int targetLocalRank = 0);

  /**
   * @brief 释放共享内存窗口资源
   */
  void freeSharedMemoryWindow();

  /**
   * @brief 节点内同步屏障
   * * 仅在 _localComm 上执行 Barrier，比全局 Barrier 快得多。
   * 用于确保内存写入完成后，其他进程再开始读取。
   */
  void sharedMemoryBarrier();

protected:
  /// Returns the communicator.
  virtual MPI_Comm &communicator(Rank rank) = 0;

  virtual Rank rank(int rank) = 0;

  // ----------------------------------------------------------------
  // [新增代码] 拓扑数据成员
  // ----------------------------------------------------------------
  MPI_Comm _localComm = MPI_COMM_NULL; // 节点内通信器
  int      _localRank = -1;            // 节点内编号
  int      _localSize = -1;            // 节点内进程数

  // ----------------------------------------------------------------
  // [新增代码] 共享内存成员变量
  // ----------------------------------------------------------------
  MPI_Win _sharedWin = MPI_WIN_NULL; // 共享内存窗口句柄
  void* _sharedBasePtr = nullptr;  // 本进程贡献的内存段起始地址 (My Segment)

  // 缓存代理进程 (Local Rank 0) 的内存地址，避免重复查询
  void* _proxySharedPtr = nullptr;

private:
  logging::Logger _log{"com::MPICommunication"};
};
} // namespace precice::com

#endif // not PRECICE_NO_MPI
