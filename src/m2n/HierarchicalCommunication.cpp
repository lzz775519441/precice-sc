#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <boost/io/ios_state.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <thread>
#include <utility>
#include <vector>
#include <cstring>
#include "m2n/HierarchicalCommunication.hpp"
#include "com/MPICommunication.hpp"
#include "m2n/SharedMemoryHeader.hpp"
#include <numeric>

#include "com/Communication.hpp"
#include "com/CommunicationFactory.hpp"
#include "com/Extra.hpp"
#include "com/Request.hpp"
#include "logging/LogMacros.hpp"
#include "m2n/DistributedCommunication.hpp"
#include "mesh/Mesh.hpp"
#include "precice/impl/Types.hpp"
#include "profiling/Event.hpp"
#include "utils/IntraComm.hpp"
#include "utils/algorithm.hpp"
#include "utils/assertion.hpp"
#include "utils/IntraComm.hpp"

using precice::profiling::Event;

namespace precice::m2n {

namespace impl {
void send(mesh::Mesh::VertexDistribution const &m,
          int                                   rankReceiver,
          const com::PtrCommunication          &communication)
{
  communication->send(static_cast<int>(m.size()), rankReceiver);

  for (auto const &i : m) {
    auto const &rank    = i.first;
    auto const &indices = i.second;
    communication->send(rank, rankReceiver);
    communication->sendRange(indices, rankReceiver);
  }
}

void receive(mesh::Mesh::VertexDistribution &m,
             int                             rankSender,
             const com::PtrCommunication    &communication)
{
  m.clear();
  int size = 0;
  communication->receive(size, rankSender);

  while (size--) {
    Rank rank = -1;
    communication->receive(rank, rankSender);
    m[rank] = communication->receiveRange(rankSender, com::asVector<int>);
  }
}

void broadcastSend(mesh::Mesh::VertexDistribution const &m,
                   const com::PtrCommunication          &communication = utils::IntraComm::getCommunication())
{
  communication->broadcast(static_cast<int>(m.size()));

  for (auto const &i : m) {
    auto const &rank    = i.first;
    auto const &indices = i.second;
    communication->broadcast(rank);
    communication->broadcast(indices);
  }
}

void broadcastReceive(mesh::Mesh::VertexDistribution &m,
                      int                             rankBroadcaster,
                      const com::PtrCommunication    &communication = utils::IntraComm::getCommunication())
{
  m.clear();
  int size = 0;
  communication->broadcast(size, rankBroadcaster);

  while (size--) {
    Rank rank = -1;
    communication->broadcast(rank, rankBroadcaster);
    communication->broadcast(m[rank], rankBroadcaster);
  }
}

void broadcast(mesh::Mesh::VertexDistribution &m)
{
  if (utils::IntraComm::isPrimary()) {
    // Broadcast (send) vertex distributions.
    m2n::impl::broadcastSend(m);
  } else if (utils::IntraComm::isSecondary()) {
    // Broadcast (receive) vertex distributions.
    m2n::impl::broadcastReceive(m, 0);
  }
}

void print(std::map<int, std::vector<int>> const &m)
{
  std::ostringstream oss;

  oss << "rank: " << utils::IntraComm::getRank() << "\n";

  for (auto &i : m) {
    for (auto &j : i.second) {
      oss << i.first << ":" << j << '\n'; // prints rank:index
    }
  }

  if (utils::IntraComm::isSecondary()) {
    utils::IntraComm::getCommunication()->send(oss.str(), 0);
  } else {

    std::string s;

    for (Rank rank : utils::IntraComm::allSecondaryRanks()) {
      utils::IntraComm::getCommunication()->receive(s, rank);

      oss << s;
    }

    std::cout << oss.str();
  }
}

void printCommunicationPartnerCountStats(std::map<int, std::vector<int>> const &m)
{
  int size = m.size();

  if (utils::IntraComm::isPrimary()) {
    size_t count   = 0;
    size_t maximum = std::numeric_limits<size_t>::min();
    size_t minimum = std::numeric_limits<size_t>::max();
    size_t total   = size;

    if (size) {
      maximum = std::max(maximum, static_cast<size_t>(size));
      minimum = std::min(minimum, static_cast<size_t>(size));
      count++;
    }

    for (Rank rank : utils::IntraComm::allSecondaryRanks()) {
      utils::IntraComm::getCommunication()->receive(size, rank);

      total += size;

      if (size) {
        maximum = std::max(maximum, static_cast<size_t>(size));
        minimum = std::min(minimum, static_cast<size_t>(size));
        count++;
      }
    }

    if (minimum > maximum)
      minimum = maximum;

    auto average = static_cast<double>(total);
    if (count != 0) {
      average /= count;
    }

    boost::io::ios_all_saver ias{std::cout};
    std::cout << std::fixed << std::setprecision(3) //
              << "Number of Communication Partners per Interface Process:"
              << "\n"
              << "  Total:   " << total << "\n"
              << "  Maximum: " << maximum << "\n"
              << "  Minimum: " << minimum << "\n"
              << "  Average: " << average << "\n"
              << "Number of Interface Processes: " << count << "\n"
              << '\n';
  } else {
    PRECICE_ASSERT(utils::IntraComm::isSecondary());
    utils::IntraComm::getCommunication()->send(size, 0);
  }
}

void printLocalIndexCountStats(std::map<int, std::vector<int>> const &m)
{
  int size = 0;

  for (auto &i : m) {
    size += i.second.size();
  }

  if (utils::IntraComm::isPrimary()) {
    size_t count   = 0;
    size_t maximum = std::numeric_limits<size_t>::min();
    size_t minimum = std::numeric_limits<size_t>::max();
    size_t total   = size;

    if (size) {
      maximum = std::max(maximum, static_cast<size_t>(size));
      minimum = std::min(minimum, static_cast<size_t>(size));

      count++;
    }

    for (Rank rank : utils::IntraComm::allSecondaryRanks()) {
      utils::IntraComm::getCommunication()->receive(size, rank);

      total += size;

      if (size) {
        maximum = std::max(maximum, static_cast<size_t>(size));
        minimum = std::min(minimum, static_cast<size_t>(size));

        count++;
      }
    }

    if (minimum > maximum)
      minimum = maximum;

    auto average = static_cast<double>(total);
    if (count != 0) {
      average /= count;
    }

    boost::io::ios_all_saver ias{std::cout};
    std::cout << std::fixed << std::setprecision(3) //
              << "Number of LVDIs per Interface Process:"
              << "\n"
              << "  Total:   " << total << '\n'
              << "  Maximum: " << maximum << '\n'
              << "  Minimum: " << minimum << '\n'
              << "  Average: " << average << '\n'
              << "Number of Interface Processes: " << count << '\n'
              << '\n';
  } else {
    PRECICE_ASSERT(utils::IntraComm::isSecondary());

    utils::IntraComm::getCommunication()->send(size, 0);
  }
}

/** builds the communication map for a local distribution given the global distribution.
 *
 *
 * @param[in] thisVertexDistribution the local vertex distribution
 * @param[in] otherVertexDistribution the total vertex distribution
 * @param[in] thisRank the rank to build the map for
 *
 * @returns the resulting communication map for rank thisRank
 *
 * The worst case complexity of the function is:
 * \f$ \mathcal{O}(p 2 (2 n)) \f$
 *
 * which consists of the computation of all intersections.
 *
 * * n is the number of data indices for each vector in `otherVertexDistribution'
 * * p number of ranks
 * * Note that n becomes smaller, if we have more ranks.
 *
 * However, in case of a proper partitioning and communication between neighbor
 * ranks (r), we would most likely end up with a factor r<<p
 * \f$ \mathcal{O}(r 2 (2 n)) \f$
 */
std::map<int, std::vector<int>> buildCommunicationMap(
    // `thisVertexDistribution' is input vertex distribution from this participant.
    mesh::Mesh::VertexDistribution const &thisVertexDistribution,
    // `otherVertexDistribution' is input vertex distribution from other participant.
    mesh::Mesh::VertexDistribution const &otherVertexDistribution,
    int                                   thisRank = utils::IntraComm::getRank())
{
  auto iterator = thisVertexDistribution.find(thisRank);
  if (iterator == thisVertexDistribution.end()) {
    return {};
  }

  std::map<int, std::vector<int>> communicationMap;
  // first a safety check, that we are actually sorted, as the function below operates
  // on sorted data sets
  PRECICE_ASSERT(std::is_sorted(iterator->second.begin(), iterator->second.end()));

  // now we iterate over all other vertex distributions to compute the intersection
  for (const auto &[rank, vertices] : otherVertexDistribution) {
    // first a safety check, that we are actually sorted, as the function below operates
    // on sorted data sets
    PRECICE_ASSERT(std::is_sorted(vertices.begin(), vertices.end()));

    // before starting to compute an actual intersection, we first check if elements can
    // possibly be in both data sets by comparing upper and lower index bounds of both
    // data sets. For typical partitioning schemes, each rank only exchanges data with
    // a few neighbors such that this check already filters out a significant amount of
    // computations
    if (iterator->second.empty() || vertices.empty() || (vertices.back() < iterator->second.at(0)) || (vertices.at(0) > iterator->second.back())) {
      // in this case there is nothing to be done
      continue;
    }
    // we have an intersection, let's compute it
    std::vector<int> inters;
    // the actual worker function, which gives us the indices of intersecting elements
    // have a look at the documentation of the function for more details
    precice::utils::set_intersection_indices(iterator->second.begin(), iterator->second.begin(), iterator->second.end(),
                                             vertices.begin(), vertices.end(),
                                             std::back_inserter(inters));
    // we have the results, now commit it into the final map
    if (!inters.empty()) {
      communicationMap.insert({rank, std::move(inters)});
    }
  }
  return communicationMap;
}

}

using namespace impl;

HierarchicalCommunication::HierarchicalCommunication(com::PtrCommunicationFactory communicationFactory,
                                                     mesh::PtrMesh                mesh)
    : DistributedCommunication(mesh),
      _communicationFactory(std::move(communicationFactory))
{
}

HierarchicalCommunication::~HierarchicalCommunication()
{
  PRECICE_TRACE(_isConnected);
  closeConnection();
}

bool HierarchicalCommunication::isConnected() const
{
  return _isConnected;
}

void HierarchicalCommunication::acceptConnection(std::string const &acceptorName,
                                                 std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  mesh::Mesh::VertexDistribution vertexDistribution = _mesh->getVertexDistribution();
  mesh::Mesh::VertexDistribution requesterVertexDistribution;

  if (not utils::IntraComm::isSecondary()) {
    PRECICE_DEBUG("Exchange vertex distribution between both primary ranks");
    Event e0("m2n.exchangeVertexDistribution");
    // Establish connection between participants' primary processes.
    auto c = _communicationFactory->newCommunication();

    c->acceptConnection(acceptorName, requesterName, "TMP-PRIMARYCOM-" + _mesh->getName(), utils::IntraComm::getRank());

    // Exchange vertex distributions.
    m2n::send(vertexDistribution, 0, c);
    m2n::receive(requesterVertexDistribution, 0, c);
  }

  PRECICE_DEBUG("Broadcast vertex distributions");
  Event e1("m2n.broadcastVertexDistributions", profiling::Synchronize);
  m2n::broadcast(vertexDistribution);
  if (utils::IntraComm::isSecondary()) {
    _mesh->setVertexDistribution(vertexDistribution);
  }
  m2n::broadcast(requesterVertexDistribution);
  e1.stop();

  // Local (for process rank in the current participant) communication map that
  // defines a mapping from a process rank in the remote participant to an array
  // of local data indices, which define a subset of local (for process rank in
  // the current participant) data to be communicated between the current
  // process rank and the remote process rank.
  //
  // Example. Assume that the current process rank is 3. Assume that its
  // `communicationMap' is
  //
  //   1 -> {1, 3}
  //   4 -> {0, 2}
  //
  // then it means that the current process (with rank 3)
  // - has to communicate (send/receive) data with local indices 1 and 3 with
  //   the remote process with rank 1;
  // - has to communicate (send/receive) data with local indices 0 and 2 with
  //   the remote process with rank 4.
  Event                           e2("m2n.buildCommunicationMap", profiling::Synchronize);
  std::map<int, std::vector<int>> communicationMap = m2n::buildCommunicationMap(
      vertexDistribution, requesterVertexDistribution);
  e2.stop();

// Print `communicationMap'.
#ifdef P2P_LCM_PRINT
  PRECICE_DEBUG("Print communication map");
  print(communicationMap);
#endif

// Print statistics of `communicationMap'.
#ifdef P2P_LCM_PRINT_STATS
  PRECICE_DEBUG("Print communication map statistics");
  printCommunicationPartnerCountStats(communicationMap);
  printLocalIndexCountStats(communicationMap);
#endif

  Event e4("m2n.createCommunications");
  e4.addData("Connections", communicationMap.size());
  if (communicationMap.empty()) {
    _isConnected = true;
    return;
  }

  PRECICE_DEBUG("Create and connect communication");
  _communication = _communicationFactory->newCommunication();

  // Accept point-to-point connections (as server) between the current acceptor
  // process (in the current participant) with rank `utils::IntraComm::getRank()'
  // and (multiple) requester processes (in the requester participant).
  _communication->acceptConnectionAsServer(acceptorName,
                                           requesterName,
                                           _mesh->getName(),
                                           utils::IntraComm::getRank(),
                                           communicationMap.size());

  PRECICE_DEBUG("Store communication map");
  for (auto const &comMap : communicationMap) {
    int  globalRequesterRank = comMap.first;
    auto indices             = std::move(communicationMap[globalRequesterRank]);

    _mappings.push_back({globalRequesterRank, std::move(indices), com::PtrRequest(), {}});
  }
  e4.stop();
  _isConnected = true;
}

void HierarchicalCommunication::acceptPreConnection(std::string const &acceptorName,
                                                    std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  const std::vector<int> &localConnectedRanks = _mesh->getConnectedRanks();

  if (localConnectedRanks.empty()) {
    _isConnected = true;
    return;
  }

  _communication = _communicationFactory->newCommunication();

  _communication->acceptConnectionAsServer(
      acceptorName,
      requesterName,
      _mesh->getName(),
      utils::IntraComm::getRank(),
      localConnectedRanks.size());

  _connectionDataVector.reserve(localConnectedRanks.size());

  for (int connectedRank : localConnectedRanks) {
    _connectionDataVector.push_back({connectedRank, com::PtrRequest()});
  }

  _isConnected = true;
}

void HierarchicalCommunication::requestConnection(std::string const &acceptorName,
                                                  std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  mesh::Mesh::VertexDistribution vertexDistribution = _mesh->getVertexDistribution();
  mesh::Mesh::VertexDistribution acceptorVertexDistribution;

  if (not utils::IntraComm::isSecondary()) {
    PRECICE_DEBUG("Exchange vertex distribution between both primary ranks");
    Event e0("m2n.exchangeVertexDistribution");
    // Establish connection between participants' primary processes.
    auto c = _communicationFactory->newCommunication();
    c->requestConnection(acceptorName, requesterName,
                         "TMP-PRIMARYCOM-" + _mesh->getName(),
                         0, 1);

    // Exchange vertex distributions.
    m2n::receive(acceptorVertexDistribution, 0, c);
    m2n::send(vertexDistribution, 0, c);
  }

  PRECICE_DEBUG("Broadcast vertex distributions");
  Event e1("m2n.broadcastVertexDistributions", profiling::Synchronize);
  m2n::broadcast(vertexDistribution);
  if (utils::IntraComm::isSecondary()) {
    _mesh->setVertexDistribution(vertexDistribution);
  }
  m2n::broadcast(acceptorVertexDistribution);
  e1.stop();

  // Local (for process rank in the current participant) communication map that
  // defines a mapping from a process rank in the remote participant to an array
  // of local data indices, which define a subset of local (for process rank in
  // the current participant) data to be communicated between the current
  // process rank and the remote process rank.
  //
  // Example. Assume that the current process rank is 3. Assume that its
  // `communicationMap' is
  //
  //   1 -> {1, 3}
  //   4 -> {0, 2}
  //
  // then it means that the current process (with rank 3)
  // - has to communicate (send/receive) data with local indices 1 and 3 with
  //   the remote process with rank 1;
  // - has to communicate (send/receive) data with local indices 0 and 2 with
  //   the remote process with rank 4.
  Event                           e2("m2n.buildCommunicationMap", profiling::Synchronize);
  std::map<int, std::vector<int>> communicationMap = m2n::buildCommunicationMap(
      vertexDistribution, acceptorVertexDistribution);
  e2.stop();

// Print `communicationMap'.
#ifdef P2P_LCM_PRINT
  PRECICE_DEBUG("Print communication map");
  print(communicationMap);
#endif

// Print statistics of `communicationMap'.
#ifdef P2P_LCM_PRINT_STATS
  PRECICE_DEBUG("Print communication map statistics");
  printCommunicationPartnerCountStats(communicationMap);
  printLocalIndexCountStats(communicationMap);
#endif

  Event e4("m2n.createCommunications");
  e4.addData("Connections", communicationMap.size());
  if (communicationMap.empty()) {
    _isConnected = true;
    return;
  }

  std::vector<com::PtrRequest> requests;
  requests.reserve(communicationMap.size());

  std::set<int> acceptingRanks;
  for (auto &i : communicationMap)
    acceptingRanks.emplace(i.first);

  PRECICE_DEBUG("Create and connect communication");
  _communication = _communicationFactory->newCommunication();
  // Request point-to-point connections (as client) between the current
  // requester process (in the current participant) and (multiple) acceptor
  // processes (in the acceptor participant) to ranks `accceptingRanks'
  // according to `communicationMap`.
  _communication->requestConnectionAsClient(acceptorName, requesterName,
                                            _mesh->getName(),
                                            acceptingRanks, utils::IntraComm::getRank());

  PRECICE_DEBUG("Store communication map");
  for (auto &i : communicationMap) {
    auto globalAcceptorRank = i.first;
    auto indices            = std::move(i.second);

    _mappings.push_back({globalAcceptorRank, std::move(indices), com::PtrRequest(), {}});
  }
  e4.stop();
  _isConnected = true;
}

void HierarchicalCommunication::requestPreConnection(std::string const &acceptorName,
                                                     std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  std::vector<int> localConnectedRanks = _mesh->getConnectedRanks();

  if (localConnectedRanks.empty()) {
    _isConnected = true;
    return;
  }

  std::vector<com::PtrRequest> requests;
  requests.reserve(localConnectedRanks.size());
  _connectionDataVector.reserve(localConnectedRanks.size());

  std::set<int> acceptingRanks(localConnectedRanks.begin(), localConnectedRanks.end());

  _communication = _communicationFactory->newCommunication();
  _communication->requestConnectionAsClient(acceptorName, requesterName,
                                            _mesh->getName(),
                                            acceptingRanks, utils::IntraComm::getRank());

  for (auto &connectedRank : localConnectedRanks) {
    _connectionDataVector.push_back({connectedRank, com::PtrRequest()});
  }
  _isConnected = true;
}

void HierarchicalCommunication::completeSecondaryRanksConnection()
{
  mesh::Mesh::CommunicationMap localCommunicationMap = _mesh->getCommunicationMap();

  for (auto &i : _connectionDataVector) {
    _mappings.push_back({i.remoteRank, std::move(localCommunicationMap[i.remoteRank]), i.request, {}});
  }
}

void HierarchicalCommunication::closeConnection()
{
  PRECICE_TRACE();

  if (not isConnected())
    return;

  checkBufferedRequests(true);

  _communication.reset();
  _mappings.clear();
  _connectionDataVector.clear();
  _isConnected = false;
}

void HierarchicalCommunication::send(precice::span<double const> itemsToSend, int valueDimension)
{


  auto mpiCom = std::dynamic_pointer_cast<com::MPICommunication>(_communication);
  PRECICE_ASSERT(mpiCom, "Hierarchical communication requires MPI.");

  // 确保上一轮的异步发送已彻底完成
  if (_isProxy) {
    waitAllOngoingRequests();
  }
  mpiCom->sharedMemoryBarrier();

  // 0. 确保路由表已建立
  if (_remoteRankToProxy.empty()) {
    exchangeTopology();
  }

  MPI_Comm localComm = mpiCom->getLocalCommunicator();
  int localSize = mpiCom->getLocalSize();
  int myGlobalRank = utils::IntraComm::getRank(); // 用于确定性排序

  // ===========================================================================
  // 步骤一：元数据申报 (Manifest)
  // ===========================================================================
  // 我们需要收集：[TargetProxy, RemoteRank, SourceRank, Bytes]
  // 使用 flat vector 存储以便 MPI 传输
  std::vector<long> myMeta;

  for (const auto &mapping : _mappings) {
    if (_remoteRankToProxy.find(mapping.remoteRank) == _remoteRankToProxy.end()) {
        continue;
    }

    long targetProxy = static_cast<long>(_remoteRankToProxy[mapping.remoteRank]);
    long remoteRank  = static_cast<long>(mapping.remoteRank);
    long sourceRank  = static_cast<long>(myGlobalRank);
    long bytes       = static_cast<long>(mapping.indices.size() * valueDimension * sizeof(double));

    if (bytes > 0) {
      myMeta.push_back(targetProxy);
      myMeta.push_back(remoteRank);
      myMeta.push_back(sourceRank);
      myMeta.push_back(bytes);
    }
  }

  // ===========================================================================
  // 步骤二：全局交换 (Allgatherv)
  // ===========================================================================
  // 1. 收集每个进程的元数据数量
  std::vector<int> metaCounts(localSize);
  int myMetaSize = static_cast<int>(myMeta.size());
  MPI_Allgather(&myMetaSize, 1, MPI_INT, metaCounts.data(), 1, MPI_INT, localComm);

  // 2. 准备接收缓冲区
  std::vector<int> metaDispls(localSize + 1, 0);
  for (int i = 0; i < localSize; ++i) {
    metaDispls[i + 1] = metaDispls[i] + metaCounts[i];
  }
  std::vector<long> globalMeta(metaDispls.back());

  // 3. 收集所有元数据
  // 注意：这里用 MPI_LONG，确保 long 是 64 位或者与 MPI_LONG 匹配
  MPI_Allgatherv(myMeta.data(), myMetaSize, MPI_LONG,
                 globalMeta.data(), metaCounts.data(), metaDispls.data(), MPI_LONG,
                 localComm);

  // ===========================================================================
  // 步骤三：排序与调度 (Sort & Schedule)
  // ===========================================================================
  // 1. 反序列化为结构体以便排序
  struct RequestEntry {
    long targetProxy;
    long remoteRank;
    long sourceRank;
    long bytes;
    long globalOffset; // 计算后回填
  };

  std::vector<RequestEntry> allRequests;
  allRequests.reserve(globalMeta.size() / 4);

  for (size_t i = 0; i < globalMeta.size(); i += 4) {
    allRequests.push_back({
      globalMeta[i],     // target
      globalMeta[i+1],   // remote
      globalMeta[i+2],   // source
      globalMeta[i+3],   // bytes
      0                  // offset (placeholder)
    });
  }

  // 2. [关键] 确定性排序
  // 优先级：TargetProxy -> RemoteRank -> SourceRank
  std::sort(allRequests.begin(), allRequests.end(),
    [](const RequestEntry &a, const RequestEntry &b) {
      if (a.targetProxy != b.targetProxy) return a.targetProxy < b.targetProxy;
      if (a.remoteRank != b.remoteRank)   return a.remoteRank < b.remoteRank;
      return a.sourceRank < b.sourceRank;
    });

  // 3. 计算内存偏移量 (Memory Layout) & 代理发送计划
  long currentShmOffset = sizeof(SharedMemoryHeader);

  // 用于 Proxy 的发送计划：TargetProxy -> {StartOffset, TotalBytes}
  struct SendBatch { long startOffset; long totalBytes; };
  std::map<long, SendBatch> sendSchedule;

  for (auto &req : allRequests) {
    req.globalOffset = currentShmOffset; // 记录这条请求的写入位置

    // 更新发送计划 (利用已排序特性，相同 Target 是连续的)
    if (sendSchedule.find(req.targetProxy) == sendSchedule.end()) {
        sendSchedule[req.targetProxy] = {currentShmOffset, 0};
    }
    sendSchedule[req.targetProxy].totalBytes += req.bytes;

    currentShmOffset += req.bytes;
  }

  // 对齐 (Optional): currentShmOffset = (currentShmOffset + 7) & ~7;
  long totalShmSize = currentShmOffset;

  // ===========================================================================
  // 步骤四：并行写入 (Parallel Write)
  // ===========================================================================
  // 申请/获取共享内存
  mpiCom->allocateSharedMemoryWindow(_isProxy ? totalShmSize : 0);
  char* basePtr = static_cast<char*>(mpiCom->getSharedMemoryPointer(0));

  // 填充 Header (仅 Proxy)
  if (_isProxy) {
     SharedMemoryHeader* header = reinterpret_cast<SharedMemoryHeader*>(basePtr);
     header->payloadSize = totalShmSize - sizeof(SharedMemoryHeader);
     header->stateFlag = 1;
  }

  // Worker 写入数据
  // 遍历排序后的列表，找到属于我的请求 (Source == MyGlobalRank)
  // 这样避免了再去 Map 里查 Offset，直接线性扫描即可
  for (const auto &req : allRequests) {
    if (req.sourceRank == myGlobalRank) {
      const Mapping* targetMapping = nullptr;
      for (const auto &m : _mappings) {
          if (static_cast<long>(m.remoteRank) == req.remoteRank) {
              targetMapping = &m;
              break;
          }
      }
      if (targetMapping) {
          double* destBuf = reinterpret_cast<double*>(basePtr + req.globalOffset);
          // 拷贝数据 (Gather)
          for (size_t i = 0; i < targetMapping->indices.size(); ++i) {
              int vertexIndex = targetMapping->indices[i];
              for (int d = 0; d < valueDimension; ++d) {
                  destBuf[i * valueDimension + d] = itemsToSend[vertexIndex * valueDimension + d];
              }
          }
      }
    }
  }

  // 节点内同步：确保所有写入完成
  mpiCom->sharedMemoryBarrier();

  // ===========================================================================
  // 步骤五：代理聚合发送 (Proxy Aggregated Send)
  // ===========================================================================
  if (_isProxy) {
    for (auto const& [targetProxy, batch] : sendSchedule) {
      if (batch.totalBytes > 0) {
        double* sendBuf = reinterpret_cast<double*>(basePtr + batch.startOffset);

        // 构造 Span 并发送
        // 注意：targetProxy 是 long，需要转回 int
        precice::span<double const> aggregatedSpan(sendBuf, batch.totalBytes / sizeof(double));
        auto req = _communication->aSend(aggregatedSpan, static_cast<int>(targetProxy));
        _ongoingRequests.push_back(std::move(req));
      }
    }
  }

  // 结束本轮同步
  mpiCom->sharedMemoryBarrier();
}

void HierarchicalCommunication::receive(precice::span<double> itemsToReceive, int valueDimension)
{

  // 按照原版逻辑，先清零缓冲区（因为后面是 += 操作）
  std::fill(itemsToReceive.begin(), itemsToReceive.end(), 0.0);

  auto mpiCom = std::dynamic_pointer_cast<com::MPICommunication>(_communication);
  PRECICE_ASSERT(mpiCom, "Hierarchical communication requires MPI.");

  // [安全关键] 在重新分配共享内存之前，必须确保上一轮（可能是 send）的异步请求已完成
  if (_isProxy) {
    waitAllOngoingRequests();
  }
  // 节点内同步：防止 Worker 还在访问旧内存，或 Proxy 还没清理完
  mpiCom->sharedMemoryBarrier();

  // 0. 确保路由表已建立
  if (_remoteRankToProxy.empty()) {
    exchangeTopology();
  }

  MPI_Comm localComm = mpiCom->getLocalCommunicator();
  int localSize = mpiCom->getLocalSize();
  int myGlobalRank = utils::IntraComm::getRank();

  // ===========================================================================
  // 步骤一：元数据申报 (Manifest)
  // ===========================================================================
  // 收集：[RemoteProxy, MyRank (Receiver), RemoteRank (Sender), Bytes]
  // 注意：这里我们声明“我期望从哪里接收多少数据”
  std::vector<long> myMeta;

  for (const auto &mapping : _mappings) {
    if (_remoteRankToProxy.find(mapping.remoteRank) == _remoteRankToProxy.end()) {
        continue;
    }

    long remoteProxy = static_cast<long>(_remoteRankToProxy[mapping.remoteRank]);
    long myRank      = static_cast<long>(myGlobalRank);
    long remoteRank  = static_cast<long>(mapping.remoteRank);
    long bytes       = static_cast<long>(mapping.indices.size() * valueDimension * sizeof(double));

    if (bytes > 0) {
      myMeta.push_back(remoteProxy);
      myMeta.push_back(myRank);
      myMeta.push_back(remoteRank);
      myMeta.push_back(bytes);
    }
  }

  // ===========================================================================
  // 步骤二：全局交换 (Allgatherv - Intra-node)
  // ===========================================================================
  // 这一步是为了让 Proxy 知道整个节点需要从外部接收多少数据，并计算布局
  std::vector<int> metaCounts(localSize);
  int myMetaSize = static_cast<int>(myMeta.size());
  MPI_Allgather(&myMetaSize, 1, MPI_INT, metaCounts.data(), 1, MPI_INT, localComm);

  std::vector<int> metaDispls(localSize + 1, 0);
  for (int i = 0; i < localSize; ++i) {
    metaDispls[i + 1] = metaDispls[i] + metaCounts[i];
  }
  std::vector<long> globalMeta(metaDispls.back());

  MPI_Allgatherv(myMeta.data(), myMetaSize, MPI_LONG,
                 globalMeta.data(), metaCounts.data(), metaDispls.data(), MPI_LONG,
                 localComm);

  // ===========================================================================
  // 步骤三：排序与调度 (Sort & Schedule)
  // ===========================================================================
  struct RequestEntry {
    long remoteProxy; // The sender proxy
    long myRank;      // Receiver rank (Me)
    long remoteRank;  // Sender rank
    long bytes;
    long globalOffset;
  };

  std::vector<RequestEntry> allRequests;
  allRequests.reserve(globalMeta.size() / 4);

  for (size_t i = 0; i < globalMeta.size(); i += 4) {
    allRequests.push_back({
      globalMeta[i],     // remoteProxy
      globalMeta[i+1],   // myRank
      globalMeta[i+2],   // remoteRank
      globalMeta[i+3],   // bytes
      0                  // offset
    });
  }

  // [关键] 确定性排序 - 必须与 Send 端的打包顺序完全匹配
  // Send 端排序: TargetProxy (Msg) -> TargetRank (Receiver) -> SourceRank (Sender)
  // Receive 端对应: RemoteProxy (Msg) -> MyRank (Receiver) -> RemoteRank (Sender)
  std::sort(allRequests.begin(), allRequests.end(),
    [](const RequestEntry &a, const RequestEntry &b) {
      if (a.remoteProxy != b.remoteProxy) return a.remoteProxy < b.remoteProxy;
      if (a.myRank != b.myRank)           return a.myRank < b.myRank;           // 对应 Send 端的 RemoteRank
      return a.remoteRank < b.remoteRank;                                       // 对应 Send 端的 SourceRank
    });

  // 计算偏移量
  long currentShmOffset = sizeof(SharedMemoryHeader);

  // 接收计划：RemoteProxy -> {StartOffset, TotalBytes}
  struct RecvBatch { long startOffset; long totalBytes; };
  std::map<long, RecvBatch> recvSchedule;

  for (auto &req : allRequests) {
    req.globalOffset = currentShmOffset;

    if (recvSchedule.find(req.remoteProxy) == recvSchedule.end()) {
        recvSchedule[req.remoteProxy] = {currentShmOffset, 0};
    }
    recvSchedule[req.remoteProxy].totalBytes += req.bytes;

    currentShmOffset += req.bytes;
  }

  long totalShmSize = currentShmOffset;

  // ===========================================================================
  // 步骤四：准备共享内存 (Allocation)
  // ===========================================================================
  // 此处会重置 Shared Memory，所以前面的 waitAllOngoingRequests 至关重要
  mpiCom->allocateSharedMemoryWindow(_isProxy ? totalShmSize : 0);
  char* basePtr = static_cast<char*>(mpiCom->getSharedMemoryPointer(0));

  if (_isProxy) {
     SharedMemoryHeader* header = reinterpret_cast<SharedMemoryHeader*>(basePtr);
     header->payloadSize = totalShmSize - sizeof(SharedMemoryHeader);
     header->stateFlag = 0; // Reset state
  }

  // ===========================================================================
  // 步骤五：代理接收 (Proxy Receive)
  // ===========================================================================
  // 此时 Proxy 负责将外部数据接收到 Shared Memory 中
  if (_isProxy) {
    std::vector<com::PtrRequest> recvRequests; // 本地暂存，这轮必须要等完

    for (auto const& [remoteProxy, batch] : recvSchedule) {
      if (batch.totalBytes > 0) {
        double* recvBuf = reinterpret_cast<double*>(basePtr + batch.startOffset);

        // 接收整个聚合包
        precice::span<double> aggregatedSpan(recvBuf, batch.totalBytes / sizeof(double));

        // 使用 aReceive 发起接收
        auto req = _communication->aReceive(aggregatedSpan, static_cast<int>(remoteProxy));
        recvRequests.push_back(std::move(req));
      }
    }

    // [阻塞等待] 接收端必须拿到数据才能分发给 Worker，所以这里不能 defer
    for (auto& req : recvRequests) {
        req->wait();
    }
  }

  // 节点内同步：Worker 等待 Proxy 接收完成
  mpiCom->sharedMemoryBarrier();

  // ===========================================================================
  // 步骤六：本地分发 (Local Scatter / Read)
  // ===========================================================================
  // Worker 遍历请求列表，找到属于自己的数据并读取
  for (const auto &req : allRequests) {
    if (req.myRank == myGlobalRank) { // 这是发给我的数据
      const Mapping* targetMapping = nullptr;

      // 找到对应的 mapping (匹配 RemoteRank)
      for (const auto &m : _mappings) {
          if (static_cast<long>(m.remoteRank) == req.remoteRank) {
              targetMapping = &m;
              break;
          }
      }

      if (targetMapping) {
          double* srcBuf = reinterpret_cast<double*>(basePtr + req.globalOffset);

          // 从共享内存读取并累加/赋值到 itemsToReceive
          // 注意：send 端是按 mapping.indices 顺序写入的，这里按相同顺序读出即可
          for (size_t i = 0; i < targetMapping->indices.size(); ++i) {
              int vertexIndex = targetMapping->indices[i];
              for (int d = 0; d < valueDimension; ++d) {
                  // 这里遵循原版逻辑使用 +=，虽然前面已经 fill 0.0
                  // 如果原意是累加（处理多重映射），这样写是安全的
                  itemsToReceive[vertexIndex * valueDimension + d] += srcBuf[i * valueDimension + d];
              }
          }
      }
    }
  }

  // 结束同步：确保所有 Worker 读完数据前，Shared Memory 不被释放/重用
  mpiCom->sharedMemoryBarrier();
}

void HierarchicalCommunication::broadcastSend(int itemToSend)
{
  for (auto &connectionData : _connectionDataVector) {
    _communication->send(itemToSend, connectionData.remoteRank);
  }
}

void HierarchicalCommunication::broadcastReceiveAll(std::vector<int> &itemToReceive)

{
  int data = 0;
  for (auto &connectionData : _connectionDataVector) {
    _communication->receive(data, connectionData.remoteRank);
    itemToReceive.push_back(data);
  }
}

void HierarchicalCommunication::broadcastSendMesh()
{
  for (auto &connectionData : _connectionDataVector) {
    com::sendMesh(*_communication, connectionData.remoteRank, *_mesh);
  }
}

void HierarchicalCommunication::broadcastReceiveAllMesh()
{
  for (auto &connectionData : _connectionDataVector) {
    com::receiveMesh(*_communication, connectionData.remoteRank, *_mesh);
  }
}

void HierarchicalCommunication::scatterAllCommunicationMap(CommunicationMap &localCommunicationMap)
{
  for (auto &connectionData : _connectionDataVector) {
    _communication->sendRange(localCommunicationMap[connectionData.remoteRank], connectionData.remoteRank);
  }
}

void HierarchicalCommunication::gatherAllCommunicationMap(CommunicationMap &localCommunicationMap)
{
  for (auto &connectionData : _connectionDataVector) {
    localCommunicationMap[connectionData.remoteRank] = _communication->receiveRange(connectionData.remoteRank, com::asVector<int>);
  }
}

void HierarchicalCommunication::checkBufferedRequests(bool blocking)
{
  PRECICE_TRACE(bufferedRequests.size());
  do {
    for (auto it = bufferedRequests.begin(); it != bufferedRequests.end();) {
      if (it->first->test())
        it = bufferedRequests.erase(it);
      else
        ++it;
    }
    if (bufferedRequests.empty())
      return;
    if (blocking)
      std::this_thread::yield(); // give up our time slice, so MPI may work
  } while (blocking);
}

std::pair<long, long> HierarchicalCommunication::computeLayout(long myItemCount, MPI_Comm localComm)
{
  // 1. 收集所有人的数据量
  // _localSize 是节点内进程总数，在构造或 split 时已获取
  std::vector<long> allCounts(_localSize);
  long myCount = myItemCount;

  // MPI_Allgather 让每个人都获得这就个数组 [10, 5, 20]
  MPI_Allgather(&myCount, 1, MPI_LONG,
                allCounts.data(), 1, MPI_LONG,
                localComm);

  // 2. 计算前缀和 (Prefix Sum)
  long totalCount = 0;
  long myOffsetCount = 0;

  for (int i = 0; i < _localSize; ++i) {
    if (i == _localRank) {
      myOffsetCount = totalCount; // 轮到我之前的累加值就是我的偏移量
    }
    totalCount += allCounts[i];
  }

  return {myOffsetCount, totalCount};
}

void HierarchicalCommunication::exchangeTopology()
{
  auto mpiCom = std::dynamic_pointer_cast<com::MPICommunication>(_communication);
  PRECICE_ASSERT(mpiCom, "Requires MPICommunication");

  if (mpiCom->getLocalCommunicator() == MPI_COMM_NULL) {
    mpiCom->splitCommunicatorByNode(MPI_COMM_WORLD);
    _localRank = mpiCom->getLocalRank();
    _localSize = mpiCom->getLocalSize();
    _isProxy   = mpiCom->isLocalRank0();
  }

  // 1. 内政：确定本地 Proxy 的全局 Rank
  int myRank = utils::IntraComm::getRank();
  int myProxyGlobalRank = 0;
  if (_isProxy) {
    myProxyGlobalRank = myRank;
  }
  MPI_Bcast(&myProxyGlobalRank, 1, MPI_INT, 0, mpiCom->getLocalCommunicator());

  // 2. 外交：与远程进程交换 Proxy 信息
  std::vector<int> targetRemoteRanks;
  for (const auto &mapping : _mappings) {
    targetRemoteRanks.push_back(mapping.remoteRank);
  }
  std::sort(targetRemoteRanks.begin(), targetRemoteRanks.end());
  targetRemoteRanks.erase(std::unique(targetRemoteRanks.begin(), targetRemoteRanks.end()), targetRemoteRanks.end());

  // [修改] 使用异步通信避免死锁
  std::vector<com::PtrRequest> requests;
  // 必须为每个远程连接准备独立的接收缓冲区，否则异步写入会冲突
  std::map<int, int> receiveBuffers;

  // A. 先发起所有接收请求 (Post Recvs)
  // 这样当对方的数据到达时，我们已经准备好接收了
  for (int remoteRank : targetRemoteRanks) {
    // 初始化 buffer (可选)
    receiveBuffers[remoteRank] = -1;
    // aReceive 返回一个 Request 对象，我们需要存下来稍后 wait
    requests.push_back(_communication->aReceive(receiveBuffers[remoteRank], remoteRank));
  }

  // B. 再发起所有发送请求 (Post Sends)
  // myProxyGlobalRank 是局部变量，但在 wait() 结束前它一直在栈上，是安全的
  for (int remoteRank : targetRemoteRanks) {
    requests.push_back(_communication->aSend(myProxyGlobalRank, remoteRank));
  }

  // C. 等待所有通信完成 (Wait All)
  for (auto &req : requests) {
    req->wait();
  }

  // 3. 建表：记录路由信息
  for (int remoteRank : targetRemoteRanks) {
    _remoteRankToProxy[remoteRank] = receiveBuffers[remoteRank];
  }

  // 确保所有人都完成了表构建
  mpiCom->sharedMemoryBarrier();
}

void HierarchicalCommunication::cleanupRequests() {
  // 移除已经完成的请求
  _ongoingRequests.erase(
    std::remove_if(_ongoingRequests.begin(), _ongoingRequests.end(),
      [](const com::PtrRequest& req) { return req->test(); }), // test() 返回 true 表示已完成
    _ongoingRequests.end());
}

void HierarchicalCommunication::waitAllOngoingRequests()
{
  if (!_ongoingRequests.empty()) {
    // 只有 Proxy 会有这些请求
    for (auto& req : _ongoingRequests) {
      req->wait(); // 阻塞等待 MPI 完成数据读取
    }
    _ongoingRequests.clear();
  }
}

} // namespace precice::m2n
