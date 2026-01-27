#ifndef PRECICE_NO_MPI

#include <cstddef>
#include <ostream>
#include <cstring>

#include "com/MPICommunication.hpp"
#include "com/MPIRequest.hpp"
#include "logging/LogMacros.hpp"
#include "precice/impl/Types.hpp"
#include "utils/span_tools.hpp"

template <size_t>
struct MPI_Select_unsigned_integer_datatype;

template <>
struct MPI_Select_unsigned_integer_datatype<1> {
  static MPI_Datatype datatype;
};
MPI_Datatype MPI_Select_unsigned_integer_datatype<1>::datatype = MPI_UNSIGNED_CHAR;

template <>
struct MPI_Select_unsigned_integer_datatype<2> {
  static MPI_Datatype datatype;
};
MPI_Datatype MPI_Select_unsigned_integer_datatype<2>::datatype = MPI_UNSIGNED_SHORT;

template <>
struct MPI_Select_unsigned_integer_datatype<4> {
  static MPI_Datatype datatype;
};
MPI_Datatype MPI_Select_unsigned_integer_datatype<4>::datatype = MPI_UNSIGNED;

template <>
struct MPI_Select_unsigned_integer_datatype<8> {
  static MPI_Datatype datatype;
};
MPI_Datatype MPI_Select_unsigned_integer_datatype<8>::datatype = MPI_UNSIGNED_LONG;

#define MPI_BOOL MPI_Select_unsigned_integer_datatype<sizeof(bool)>::datatype

namespace precice::com {
MPICommunication::MPICommunication() = default;

MPICommunication::~MPICommunication()
{
  // 如果创建了本地通信器，销毁它以释放资源
  if (_localComm != MPI_COMM_NULL) {
    MPI_Comm_free(&_localComm);
    _localComm = MPI_COMM_NULL;
  }
}

void MPICommunication::send(std::string const &itemToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemToSend, rankReceiver);
  rankReceiver = adjustRank(rankReceiver);
  PRECICE_DEBUG("Message: " + itemToSend);
  MPI_Send(const_cast<char *>(itemToSend.c_str()),
           itemToSend.size(),
           MPI_CHAR,
           rank(rankReceiver),
           0,
           communicator(rankReceiver));
}

void MPICommunication::send(precice::span<const int> itemsToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemsToSend.size());
  rankReceiver = adjustRank(rankReceiver);
  MPI_Send(const_cast<int *>(itemsToSend.data()),
           itemsToSend.size(),
           MPI_INT,
           rank(rankReceiver),
           0,
           communicator(rankReceiver));
}

PtrRequest MPICommunication::aSend(precice::span<const int> itemsToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemsToSend.size());
  rankReceiver = adjustRank(rankReceiver);

  MPI_Request request;
  MPI_Isend(const_cast<int *>(itemsToSend.data()),
            itemsToSend.size(),
            MPI_INT,
            rank(rankReceiver),
            0,
            communicator(rankReceiver),
            &request);

  return PtrRequest(new MPIRequest(request));
}

void MPICommunication::send(precice::span<const double> itemsToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemsToSend.size());
  rankReceiver = adjustRank(rankReceiver);
  MPI_Send(const_cast<double *>(itemsToSend.data()),
           itemsToSend.size(),
           MPI_DOUBLE,
           rank(rankReceiver),
           0,
           communicator(rankReceiver));
}

PtrRequest MPICommunication::aSend(precice::span<const double> itemsToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemsToSend.size(), rankReceiver);
  rankReceiver = adjustRank(rankReceiver);

  MPI_Request request;
  MPI_Isend(const_cast<double *>(itemsToSend.data()),
            itemsToSend.size(),
            MPI_DOUBLE,
            rank(rankReceiver),
            0,
            communicator(rankReceiver),
            &request);

  return PtrRequest(new MPIRequest(request));
}

void MPICommunication::send(double itemToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemToSend, rankReceiver);
  rankReceiver = adjustRank(rankReceiver);
  MPI_Send(&itemToSend,
           1,
           MPI_DOUBLE,
           rank(rankReceiver),
           0,
           communicator(rankReceiver));
}

PtrRequest MPICommunication::aSend(const double &itemToSend, Rank rankReceiver)
{
  return aSend(precice::refToSpan<const double>(itemToSend), rankReceiver);
}

void MPICommunication::send(int itemToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemToSend, rankReceiver);
  rankReceiver = adjustRank(rankReceiver);
  MPI_Send(&itemToSend,
           1,
           MPI_INT,
           rank(rankReceiver),
           0,
           communicator(rankReceiver));
}

PtrRequest MPICommunication::aSend(const int &itemToSend, Rank rankReceiver)
{
  return aSend(precice::refToSpan<const int>(itemToSend), rankReceiver);
}

void MPICommunication::send(bool itemToSend, Rank rankReceiver)
{
  PRECICE_TRACE(itemToSend, rankReceiver);
  rankReceiver = adjustRank(rankReceiver);
  MPI_Send(&itemToSend,
           1,
           MPI_BOOL,
           rank(rankReceiver),
           0,
           communicator(rankReceiver));
}

PtrRequest MPICommunication::aSend(const bool &itemToSend, Rank rankReceiver)
{
  PRECICE_TRACE();
  rankReceiver = adjustRank(rankReceiver);

  MPI_Request request;
  MPI_Isend(const_cast<bool *>(&itemToSend),
            1,
            MPI_BOOL,
            rank(rankReceiver),
            0,
            communicator(rankReceiver),
            &request);

  return PtrRequest(new MPIRequest(request));
}

void MPICommunication::receive(std::string &itemToReceive, Rank rankSender)
{
  PRECICE_TRACE(itemToReceive, rankSender);
  rankSender = adjustRank(rankSender);
  int        length;
  MPI_Status status;
  MPI_Probe(rank(rankSender), 0, communicator(rankSender), &status);
  MPI_Get_count(&status, MPI_CHAR, &length);
  PRECICE_DEBUG("Stringlength = {}", length);
  itemToReceive = std::string(length, '\0');
  MPI_Recv(const_cast<char *>(itemToReceive.data()),
           length,
           MPI_CHAR,
           rank(rankSender),
           0,
           communicator(rankSender),
           MPI_STATUS_IGNORE);
  PRECICE_DEBUG("Received \"{}\" from rank {}", itemToReceive, rankSender);
}

void MPICommunication::receive(precice::span<int> itemsToReceive, Rank rankSender)
{
  PRECICE_TRACE(itemsToReceive.size());
  rankSender = adjustRank(rankSender);

  MPI_Status status;
  MPI_Recv(itemsToReceive.data(),
           itemsToReceive.size(),
           MPI_INT,
           rank(rankSender),
           0,
           communicator(rankSender),
           &status);
}

void MPICommunication::receive(precice::span<double> itemsToReceive, Rank rankSender)
{
  PRECICE_TRACE(itemsToReceive.size());
  rankSender = adjustRank(rankSender);

  MPI_Status status;
  MPI_Recv(itemsToReceive.data(),
           itemsToReceive.size(),
           MPI_DOUBLE,
           rank(rankSender),
           0,
           communicator(rankSender),
           &status);
}

PtrRequest MPICommunication::aReceive(precice::span<double> itemsToReceive, Rank rankSender)
{
  PRECICE_TRACE(itemsToReceive.size());
  rankSender = adjustRank(rankSender);

  MPI_Request request;
  MPI_Irecv(itemsToReceive.data(),
            itemsToReceive.size(),
            MPI_DOUBLE,
            rank(rankSender),
            0,
            communicator(rankSender),
            &request);

  return PtrRequest(new MPIRequest(request));
}

void MPICommunication::receive(double &itemToReceive, Rank rankSender)
{
  PRECICE_TRACE(rankSender);
  rankSender = adjustRank(rankSender);

  MPI_Status status;
  MPI_Recv(&itemToReceive,
           1,
           MPI_DOUBLE,
           rank(rankSender),
           0,
           communicator(rankSender),
           &status);
  PRECICE_DEBUG("Received {} from rank {}", itemToReceive, rankSender);
}

PtrRequest MPICommunication::aReceive(double &itemToReceive, Rank rankSender)
{
  return aReceive(precice::refToSpan<double>(itemToReceive), rankSender);
}

void MPICommunication::receive(int &itemToReceive, Rank rankSender)
{
  PRECICE_TRACE(rankSender);
  rankSender = adjustRank(rankSender);

  MPI_Status status;
  MPI_Recv(&itemToReceive,
           1,
           MPI_INT,
           rank(rankSender),
           0,
           communicator(rankSender),
           &status);
  PRECICE_DEBUG("Received {} from rank {}", itemToReceive, rankSender);
}

PtrRequest MPICommunication::aReceive(int &itemToReceive, Rank rankSender)
{
  PRECICE_TRACE(rankSender);
  rankSender = adjustRank(rankSender);

  MPI_Request request;
  MPI_Irecv(&itemToReceive,
            1,
            MPI_INT,
            rank(rankSender),
            0,
            communicator(rankSender),
            &request);

  return PtrRequest(new MPIRequest(request));
}

void MPICommunication::receive(bool &itemToReceive, Rank rankSender)
{
  PRECICE_TRACE(rankSender);
  rankSender = adjustRank(rankSender);

  MPI_Status status;
  MPI_Recv(&itemToReceive,
           1,
           MPI_BOOL,
           rank(rankSender),
           0,
           communicator(rankSender),
           &status);
  PRECICE_DEBUG("Received {} from rank {}", itemToReceive, rankSender);
}

PtrRequest MPICommunication::aReceive(bool &itemToReceive, Rank rankSender)
{
  PRECICE_TRACE(rankSender);
  rankSender = adjustRank(rankSender);

  MPI_Request request;
  MPI_Irecv(&itemToReceive,
            1,
            MPI_BOOL,
            rank(rankSender),
            0,
            communicator(rankSender),
            &request);

  return PtrRequest(new MPIRequest(request));
}

// ----------------------------------------------------------------
// [新增代码] 拓扑感知具体实现
// ----------------------------------------------------------------

void MPICommunication::splitCommunicatorByNode(MPI_Comm globalComm)
{
  // 防止重复初始化
  if (_localComm != MPI_COMM_NULL) {
    return;
  }

  // MPI_COMM_TYPE_SHARED 是 MPI-3 标准引入的
  // 它会将 globalComm 分裂，使得共享同一块内存（即同一物理节点）的进程进入同一个子通信器
  // key=0 表示我们希望保持原有的 Rank 相对顺序
  int mpiResult = MPI_Comm_split_type(globalComm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &_localComm);

  if (mpiResult != MPI_SUCCESS) {
    PRECICE_ERROR("Failed to split MPI communicator for shared memory topology discovery.");
    return;
  }

  // 获取在节点内部的 Rank 和 Size
  MPI_Comm_rank(_localComm, &_localRank);
  MPI_Comm_size(_localComm, &_localSize);

  PRECICE_DEBUG("Topology Discovery: Local Rank = {}, Local Size = {} on this physical node.", _localRank, _localSize);

  if (isLocalRank0()) {
    PRECICE_DEBUG("I am the Proxy Agent (Local Rank 0) for this node.");
  }
}

int MPICommunication::getLocalRank() const
{
  return _localRank;
}

int MPICommunication::getLocalSize() const
{
  return _localSize;
}

bool MPICommunication::isLocalRank0() const
{
  // 约定：节点内的第 0 号进程作为代理
  return _localRank == 0;
}

MPI_Comm MPICommunication::getLocalCommunicator() const
{
  return _localComm;
}

// ----------------------------------------------------------------
// [新增代码] 共享内存管理实现 (第二步)
// ----------------------------------------------------------------

void MPICommunication::sharedMemoryBarrier()
{
  if (_localComm != MPI_COMM_NULL) {
    MPI_Barrier(_localComm);
  }
}

} // namespace precice::com

#endif // not PRECICE_NO_MPI
