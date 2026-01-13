#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "com/Request.hpp"
#include "DistributedCommunication.hpp"
#include "com/SharedPointer.hpp"
#include "logging/Logger.hpp"
#include "mesh/Mesh.hpp"
#include "mesh/SharedPointer.hpp"
#include "m2n/SharedMemoryHeader.hpp"
#include "com/MPICommunication.hpp"
#include <map>
#include <mpi.h>

namespace precice {
namespace com {
class Request;
} // namespace com

namespace m2n {
/**
 * @brief Point-to-point communication implementation of DistributedCommunication.
 *
 * Direct communication of local data subsets is performed between processes of
 * coupled participants. The two supported implementations of direct
 * communication are SocketCommunication and MPIPortsCommunication which can be
 * supplied via their corresponding instantiation factories
 * SocketCommunicationFactory and MPIPortsCommunicationFactory.
 *
 * For the detailed implementation documentation refer to HierarchicalCommunication.cpp.
 */
class HierarchicalCommunication : public DistributedCommunication {
public:
  HierarchicalCommunication(com::PtrCommunicationFactory communicationFactory,
                            mesh::PtrMesh                mesh);

  ~HierarchicalCommunication() override;

  /// Returns true, if a connection to a remote participant has been established.
  bool isConnected() const override;

  /**
   * @brief Accepts connection from participant, which has to call
   *        requestConnection().
   *
   * @param[in] acceptorName  Name of calling participant.
   * @param[in] requesterName Name of remote participant to connect to.
   */
  void acceptConnection(std::string const &acceptorName,
                        std::string const &requesterName) override;

  /**
   * @brief Requests connection from participant, which has to call acceptConnection().
   *
   * @param[in] acceptorName Name of remote participant to connect to.
   * @param[in] requesterName Name of calling participant.
   */
  void requestConnection(std::string const &acceptorName,
                         std::string const &requesterName) override;

  /**
   * @brief Accepts connection from participant, which has to call
   *        requestPreConnection().
   *        Only initial connection is created.
   *
   * @param[in] acceptorName  Name of calling participant.
   * @param[in] requesterName Name of remote participant to connect to.
   */
  void acceptPreConnection(std::string const &acceptorName,
                           std::string const &requesterName) override;

  /**
   * @brief Requests connection from participant, which has to call acceptConnection().
   *        Only initial connection is created.
   *
   * @param[in] acceptorName Name of remote participant to connect to.
   * @param[in] requesterName Name of calling participant.
   */
  void requestPreConnection(std::string const &acceptorName,
                            std::string const &requesterName) override;

  /// Completes the secondary connections for both acceptor and requester by updating the vertex list in _mappings
  void completeSecondaryRanksConnection() override;

  /**
   * @brief Disconnects from communication space, i.e. participant.
   *
   * This method is called on destruction.
   */
  void closeConnection() override;

  /**
   * @brief Sends a subset of local double values corresponding to local indices
   *        deduced from the current and remote vertex distributions.
   */
  void send(precice::span<double const> itemsToSend, int valueDimension = 1) override;

  /**
   * @brief Receives a subset of local double values corresponding to local
   *        indices deduced from the current and remote vertex distributions.
   */
  void receive(precice::span<double> itemsToReceive, int valueDimension = 1) override;

  /// Broadcasts an int to connected ranks on remote participant
  void broadcastSend(int itemToSend) override;

  /**
   * @brief Receives an int per connected rank on remote participant
   * @para[out] itemToReceive received ints from remote ranks are stored with the sender rank order
   */
  void broadcastReceiveAll(std::vector<int> &itemToReceive) override;

  /// Broadcasts a mesh to connected ranks on remote participant
  void broadcastSendMesh() override;

  /// Receive mesh partitions per connected rank on remote participant
  void broadcastReceiveAllMesh() override;

  /// Scatters a communication map over connected ranks on remote participant
  void scatterAllCommunicationMap(CommunicationMap &localCommunicationMap) override;

  /// Gathers a communication maps from connected ranks on remote participant
  void gatherAllCommunicationMap(CommunicationMap &localCommunicationMap) override;

private:
  logging::Logger _log{"m2n::HierarchicalCommunication"};

  /// Checks all stored requests for completion and removes associated buffers
  /**
   * @param[in] blocking False means that the function returns, even when there are requests left.
   */
  void checkBufferedRequests(bool blocking);

  com::PtrCommunicationFactory _communicationFactory;

  /// Communication class used for this HierarchicalCommunication
  /**
   * A Communication object represents all connections to all ranks made by this P2P instance.
   **/
  com::PtrCommunication _communication;

  /**
   * @brief Defines mapping between:
   *        1. global remote process rank;
   *        2. local data indices, which define a subset of local (for process
   *           rank in the current participant) data to be communicated between
   *           the current process rank and the remote process rank;
   *        3. Request holding information about pending communication
   *        4. Appropriately sized buffer to receive elements
   */
  struct Mapping {
    int                 remoteRank;
    std::vector<int>    indices;
    com::PtrRequest     request;
    std::vector<double> recvBuffer;
  };

  /**
   * @brief Local (for process rank in the current participant) vector of
   *        mappings (one to service each point-to-point connection).
   */
  std::vector<Mapping> _mappings;

  /**
   * @brief this data structure is used to store m2n communication information for the 1 step of
   *        bounding box initialization. It stores:
   *        1. global remote process rank;
   *        2. communication object (provides point-to-point communication routines).
   *        3. Request holding information about pending communication
   */
  struct ConnectionData {
    int             remoteRank;
    com::PtrRequest request;
  };

  /**
   * @brief Local (for process rank in the current participant) vector of
   *        ConnectionData (one to service each point-to-point connection).
   */
  std::vector<ConnectionData> _connectionDataVector;

  bool _isConnected = false;

  std::list<std::pair<std::shared_ptr<com::Request>,
                      std::shared_ptr<std::vector<double>>>>
      bufferedRequests;

  // [新增] 拓扑信息
  bool _isProxy = false;
  int  _localRank = -1;
  int  _localSize = -1;

  // 路由表：远程 Rank -> 远程 Proxy Rank
  std::map<int, int> _remoteRankToProxy;

  // 辅助函数：执行拓扑交换
  void exchangeTopology();

  std::vector<com::PtrRequest> _ongoingRequests;
  void cleanupRequests();
  void waitAllOngoingRequests();

  // 定义任务结构体 (Send/Recv 通用)
  struct ProxyTask {
    int targetRank;    // 对方的 Rank
    long shmOffset;    // 在对应 Window 中的偏移量
    long totalDoubles; // 数据量
  };

  struct WorkerTask {
    double* shmPtr;        // 绝对地址
    const int* indicesPtr; // 映射索引
    size_t count;          // 数量
  };

  // ---------------------------------------------------------
  // 1. 发送端专用 (SEND Resource)
  // ---------------------------------------------------------
  int     _cachedSendDim = 0;       // 记录当前的 Send 维度
  MPI_Win _winSend = MPI_WIN_NULL;  // 发送专用 MPI Window
  char* _sendBasePtr = nullptr;   // 发送内存基地址

  std::vector<ProxyTask>  _proxySendTasks;
  std::vector<WorkerTask> _workerSendTasks;

  void initializeSendPattern(int valueDimension);
  void freeSendWindow();

  // ---------------------------------------------------------
  // 2. 接收端专用 (RECEIVE Resource)
  // ---------------------------------------------------------
  int     _cachedRecvDim = 0;       // 记录当前的 Recv 维度
  MPI_Win _winRecv = MPI_WIN_NULL;  // 接收专用 MPI Window
  char* _recvBasePtr = nullptr;   // 接收内存基地址

  std::vector<ProxyTask>  _proxyRecvTasks;
  std::vector<WorkerTask> _workerRecvTasks;

  void initializeRecvPattern(int valueDimension);
  void freeRecvWindow();
};
} // namespace m2n
} // namespace precice
