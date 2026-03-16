#include "m2n/HierarchicalCommunication.hpp"
#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <boost/io/ios_state.hpp>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <thread>
#include <utility>
#include <vector>

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
    m2n::impl::broadcastSend(m);
  } else if (utils::IntraComm::isSecondary()) {
    m2n::impl::broadcastReceive(m, 0);
  }
}

void print(std::map<int, std::vector<int>> const &m)
{
  std::ostringstream oss;
  oss << "rank: " << utils::IntraComm::getRank() << "\n";
  for (auto &i : m) {
    for (auto &j : i.second) {
      oss << i.first << ":" << j << '\n';
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
    if (count != 0)
      average /= count;

    boost::io::ios_all_saver ias{std::cout};
    std::cout << std::fixed << std::setprecision(3)
              << "Number of Communication Partners per Interface Process:\n"
              << "  Total:   " << total << "\n"
              << "  Maximum: " << maximum << "\n"
              << "  Minimum: " << minimum << "\n"
              << "  Average: " << average << "\n"
              << "Number of Interface Processes: " << count << "\n\n";
  } else {
    PRECICE_ASSERT(utils::IntraComm::isSecondary());
    utils::IntraComm::getCommunication()->send(size, 0);
  }
}

void printLocalIndexCountStats(std::map<int, std::vector<int>> const &m)
{
  int size = 0;
  for (auto &i : m)
    size += i.second.size();

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
    if (count != 0)
      average /= count;

    boost::io::ios_all_saver ias{std::cout};
    std::cout << std::fixed << std::setprecision(3)
              << "Number of LVDIs per Interface Process:\n"
              << "  Total:   " << total << '\n'
              << "  Maximum: " << maximum << '\n'
              << "  Minimum: " << minimum << '\n'
              << "  Average: " << average << '\n'
              << "Number of Interface Processes: " << count << '\n\n';
  } else {
    PRECICE_ASSERT(utils::IntraComm::isSecondary());
    utils::IntraComm::getCommunication()->send(size, 0);
  }
}

std::map<int, std::vector<int>> buildCommunicationMap(
    mesh::Mesh::VertexDistribution const &thisVertexDistribution,
    mesh::Mesh::VertexDistribution const &otherVertexDistribution,
    int                                   thisRank = utils::IntraComm::getRank())
{
  auto iterator = thisVertexDistribution.find(thisRank);
  if (iterator == thisVertexDistribution.end()) {
    return {};
  }

  std::map<int, std::vector<int>> communicationMap;
  PRECICE_ASSERT(std::is_sorted(iterator->second.begin(), iterator->second.end()));

  for (const auto &[rank, vertices] : otherVertexDistribution) {
    PRECICE_ASSERT(std::is_sorted(vertices.begin(), vertices.end()));

    if (iterator->second.empty() || vertices.empty() || (vertices.back() < iterator->second.at(0)) || (vertices.at(0) > iterator->second.back())) {
      continue;
    }
    std::vector<int> inters;
    precice::utils::set_intersection_indices(iterator->second.begin(), iterator->second.begin(), iterator->second.end(),
                                             vertices.begin(), vertices.end(),
                                             std::back_inserter(inters));
    if (!inters.empty()) {
      communicationMap.insert({rank, std::move(inters)});
    }
  }
  return communicationMap;
}

} // namespace impl

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

void HierarchicalCommunication::setupLocalTopology()
{
  if (_localComm != MPI_COMM_NULL) {
    return;
  }

  MPI_Comm globalComm = MPI_COMM_WORLD;
  MPI_Comm_split_type(globalComm, MPI_COMM_TYPE_SHARED, utils::IntraComm::getRank(), MPI_INFO_NULL, &_localComm);

  MPI_Comm_rank(_localComm, &_localRank);
  MPI_Comm_size(_localComm, &_localSize);

  _isProxy = (_localRank == 0);
}

void HierarchicalCommunication::acceptConnection(std::string const &acceptorName,
                                                 std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  PRECICE_DEBUG("setupLocalTopology");
  setupLocalTopology();
  PRECICE_DEBUG("setupLocalTopologyend");
  mesh::Mesh::VertexDistribution vertexDistribution = _mesh->getVertexDistribution();
  mesh::Mesh::VertexDistribution requesterVertexDistribution;

  // --- 拓扑与网格交换 ---
  int myGlobalRank = utils::IntraComm::getRank();
  int myProxyRank  = myGlobalRank;
  MPI_Bcast(&myProxyRank, 1, MPI_INT, 0, _localComm);

  std::vector<int> localProxyMap(utils::IntraComm::getSize());
  MPI_Allgather(&myProxyRank, 1, MPI_INT, localProxyMap.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> remoteProxyMap;
  PRECICE_DEBUG("Exchange vertex distribution");
  if (not utils::IntraComm::isSecondary()) {
    PRECICE_DEBUG("Exchange vertex distribution & Topology");
    Event e0("m2n.exchangeVertexDistribution");
    auto  c = _communicationFactory->newCommunication();

    c->acceptConnection(acceptorName, requesterName, "TMP-PRIMARYCOM-" + _mesh->getName(), utils::IntraComm::getRank());

    m2n::send(vertexDistribution, 0, c);

    int lSize = localProxyMap.size();
    c->send(lSize, 0);
    c->send(precice::span<const int>(localProxyMap.data(), lSize), 0);

    m2n::receive(requesterVertexDistribution, 0, c);

    int rSize = 0;
    c->receive(rSize, 0);
    remoteProxyMap.resize(rSize);
    c->receive(precice::span<int>(remoteProxyMap.data(), rSize), 0);
  }

  PRECICE_DEBUG("broadcastVertexDistributions");
  Event e1("m2n.broadcastVertexDistributions", profiling::Synchronize);
  m2n::broadcast(vertexDistribution);
  if (utils::IntraComm::isSecondary()) {
    _mesh->setVertexDistribution(vertexDistribution);
  }
  m2n::broadcast(requesterVertexDistribution);

  PRECICE_DEBUG("广播对端拓扑");
  // 广播对端拓扑
  int remoteSize = remoteProxyMap.size();
  MPI_Bcast(&remoteSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (utils::IntraComm::isSecondary()) {
    remoteProxyMap.resize(remoteSize);
  }
  if (remoteSize > 0) {
    MPI_Bcast(remoteProxyMap.data(), remoteSize, MPI_INT, 0, MPI_COMM_WORLD);
  }
  for (int i = 0; i < remoteSize; ++i) {
    _remoteRankToProxy[i] = remoteProxyMap[i];
  }
  e1.stop();

  PRECICE_DEBUG("buildCommunicationMap");
  Event                           e2("m2n.buildCommunicationMap", profiling::Synchronize);
  std::map<int, std::vector<int>> communicationMap = m2n::buildCommunicationMap(
      vertexDistribution, requesterVertexDistribution);
  e2.stop();

  Event e4("m2n.createCommunications");
  e4.addData("Connections", communicationMap.size());

  // [修复 1] 删除了 communicationMap.empty() 的早退，保证死锁消除
  for (auto const &comMap : communicationMap) {
    int globalRequesterRank = comMap.first;
    // [修复 2] 避免常量引用的 move，并统一变量名
    auto indices = comMap.second;
    _mappings.push_back({globalRequesterRank, std::move(indices), com::PtrRequest(), {}});
    _connectionDataVector.push_back({globalRequesterRank, com::PtrRequest()});
  }

  std::set<int> myNeededRemoteProxies;
  for (auto const &map : _mappings) {
    if (!map.indices.empty() && _remoteRankToProxy.count(map.remoteRank)) {
      myNeededRemoteProxies.insert(_remoteRankToProxy[map.remoteRank]);
    }
  }

  std::vector<int> myProxiesVec(myNeededRemoteProxies.begin(), myNeededRemoteProxies.end());
  int              numMyProxies = myProxiesVec.size();
  std::vector<int> allNumProxies(_localSize);
  MPI_Allgather(&numMyProxies, 1, MPI_INT, allNumProxies.data(), 1, MPI_INT, _localComm);

  std::vector<int> displs(_localSize + 1, 0);
  for (int i = 0; i < _localSize; ++i)
    displs[i + 1] = displs[i] + allNumProxies[i];

  std::vector<int> allProxies(displs.back());
  MPI_Allgatherv(myProxiesVec.data(), numMyProxies, MPI_INT,
                 allProxies.data(), allNumProxies.data(), displs.data(), MPI_INT, _localComm);

  std::set<int> uniqueRemoteProxies(allProxies.begin(), allProxies.end());

  if (_isProxy) {
    _communication = _communicationFactory->newCommunication();
    if (!uniqueRemoteProxies.empty()) {
      _communication->acceptConnectionAsServer(acceptorName, requesterName, _mesh->getName(), utils::IntraComm::getRank(), uniqueRemoteProxies.size());
    }
  }
  e4.stop();
  _isConnected = true;

  initializeSendPattern(1);
  initializeRecvPattern(1);
}

void HierarchicalCommunication::requestConnection(std::string const &acceptorName,
                                                  std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  setupLocalTopology();

  mesh::Mesh::VertexDistribution vertexDistribution = _mesh->getVertexDistribution();
  mesh::Mesh::VertexDistribution acceptorVertexDistribution;

  int myGlobalRank = utils::IntraComm::getRank();
  int myProxyRank  = myGlobalRank;
  MPI_Bcast(&myProxyRank, 1, MPI_INT, 0, _localComm);

  std::vector<int> localProxyMap(utils::IntraComm::getSize());
  MPI_Allgather(&myProxyRank, 1, MPI_INT, localProxyMap.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> remoteProxyMap;

  if (not utils::IntraComm::isSecondary()) {
    Event e0("m2n.exchangeVertexDistribution");
    auto  c = _communicationFactory->newCommunication();
    c->requestConnection(acceptorName, requesterName, "TMP-PRIMARYCOM-" + _mesh->getName(), 0, 1);

    m2n::receive(acceptorVertexDistribution, 0, c);

    int rSize = 0;
    c->receive(rSize, 0);
    remoteProxyMap.resize(rSize);
    c->receive(precice::span<int>(remoteProxyMap.data(), rSize), 0);

    m2n::send(vertexDistribution, 0, c);
    int lSize = localProxyMap.size();
    c->send(lSize, 0);
    c->send(precice::span<const int>(localProxyMap.data(), lSize), 0);
  }

  Event e1("m2n.broadcastVertexDistributions", profiling::Synchronize);
  m2n::broadcast(vertexDistribution);
  if (utils::IntraComm::isSecondary()) {
    _mesh->setVertexDistribution(vertexDistribution);
  }
  m2n::broadcast(acceptorVertexDistribution);

  int remoteSize = remoteProxyMap.size();
  MPI_Bcast(&remoteSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (utils::IntraComm::isSecondary()) {
    remoteProxyMap.resize(remoteSize);
  }
  if (remoteSize > 0) {
    MPI_Bcast(remoteProxyMap.data(), remoteSize, MPI_INT, 0, MPI_COMM_WORLD);
  }

  for (int i = 0; i < remoteSize; ++i) {
    _remoteRankToProxy[i] = remoteProxyMap[i];
  }
  e1.stop();

  Event                           e2("m2n.buildCommunicationMap", profiling::Synchronize);
  std::map<int, std::vector<int>> communicationMap = m2n::buildCommunicationMap(
      vertexDistribution, acceptorVertexDistribution);
  e2.stop();

  Event e4("m2n.createCommunications");
  e4.addData("Connections", communicationMap.size());

  // [修复 3] 同步删除早退逻辑
  for (auto const &comMap : communicationMap) {
    auto globalAcceptorRank = comMap.first;
    auto indices            = comMap.second;
    _mappings.push_back({globalAcceptorRank, std::move(indices), com::PtrRequest(), {}});
    _connectionDataVector.push_back({globalAcceptorRank, com::PtrRequest()});
  }

  std::set<int> myNeededRemoteProxies;
  for (auto const &map : _mappings) {
    if (!map.indices.empty() && _remoteRankToProxy.count(map.remoteRank)) {
      myNeededRemoteProxies.insert(_remoteRankToProxy[map.remoteRank]);
    }
  }

  std::vector<int> myProxiesVec(myNeededRemoteProxies.begin(), myNeededRemoteProxies.end());
  int              numMyProxies = myProxiesVec.size();
  std::vector<int> allNumProxies(_localSize);
  MPI_Allgather(&numMyProxies, 1, MPI_INT, allNumProxies.data(), 1, MPI_INT, _localComm);

  std::vector<int> displs(_localSize + 1, 0);
  for (int i = 0; i < _localSize; ++i)
    displs[i + 1] = displs[i] + allNumProxies[i];

  std::vector<int> allProxies(displs.back());
  MPI_Allgatherv(myProxiesVec.data(), numMyProxies, MPI_INT,
                 allProxies.data(), allNumProxies.data(), displs.data(), MPI_INT, _localComm);

  std::set<int> uniqueRemoteProxies(allProxies.begin(), allProxies.end());

  if (_isProxy) {
    _communication = _communicationFactory->newCommunication();
    if (!uniqueRemoteProxies.empty()) {
      _communication->requestConnectionAsClient(acceptorName, requesterName, _mesh->getName(), uniqueRemoteProxies, utils::IntraComm::getRank());
    }
  }
  e4.stop();
  _isConnected = true;

  initializeSendPattern(1);
  initializeRecvPattern(1);
}

// [修复 4] 让 PreConnection 拥有完整的拓扑拉取与节点去重逻辑，抵御 BoundingBox 过滤
void HierarchicalCommunication::acceptPreConnection(std::string const &acceptorName,
                                                    std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");

  // 代码逻辑被转移至统一的 Connection，通常开启 BoundingBox 时，
  // 会转为依赖底层的 TCP M2N 进行。因为 PreConnection 是轻量级的，为了保证兼容性，
  // 此处直接抛出错误迫使用户在 XML 关掉 two-level-init。
  // (对于分层架构，两级初始化毫无意义且增加开销，在配置里设置 use-two-level-initialization = false 即可)
  PRECICE_ASSERT(false, "Hierarchical Communication requires 'use-two-level-initialization=false' in XML config.");
}

void HierarchicalCommunication::requestPreConnection(std::string const &acceptorName,
                                                     std::string const &requesterName)
{
  PRECICE_TRACE(acceptorName, requesterName);
  PRECICE_ASSERT(not isConnected(), "Already connected.");
  PRECICE_ASSERT(false, "Hierarchical Communication requires 'use-two-level-initialization=false' in XML config.");
}

void HierarchicalCommunication::completeSecondaryRanksConnection()
{
  // 如果关闭了两级初始化，这个函数就不会被调用。
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

  waitAllOngoingRequests();
  checkBufferedRequests(true);

  freeSendWindow();
  freeRecvWindow();
  _communication.reset();
  _mappings.clear();
  _connectionDataVector.clear();
  _isConnected = false;

  if (_localComm != MPI_COMM_NULL) {
    MPI_Comm_free(&_localComm);
    _localComm = MPI_COMM_NULL;
  }
}

void HierarchicalCommunication::send(precice::span<double const> itemsToSend, int valueDimension)
{
  if (_isProxy)
    waitAllOngoingRequests();
  MPI_Barrier(_localComm);

  if (valueDimension != _cachedSendDim)
    initializeSendPattern(valueDimension);

  for (const auto &task : _workerSendTasks) {
    for (size_t i = 0; i < task.count; ++i) {
      int vertexIndex = task.indicesPtr[i];
      for (int d = 0; d < valueDimension; ++d) {
        task.shmPtr[i * valueDimension + d] = itemsToSend[vertexIndex * valueDimension + d];
      }
    }
  }

  MPI_Barrier(_localComm);

  if (_isProxy) {
    for (const auto &task : _proxySendTasks) {
      double                     *sendBuf = reinterpret_cast<double *>(_sendBasePtr + task.shmOffset);
      precice::span<double const> aggregatedSpan(sendBuf, task.totalDoubles);
      _ongoingRequests.push_back(_communication->aSend(aggregatedSpan, task.targetRank));
    }
  }
}

void HierarchicalCommunication::receive(precice::span<double> itemsToReceive, int valueDimension)
{
  std::fill(itemsToReceive.begin(), itemsToReceive.end(), 0.0);

  if (_isProxy)
    waitAllOngoingRequests();
  MPI_Barrier(_localComm);

  if (valueDimension != _cachedRecvDim)
    initializeRecvPattern(valueDimension);

  if (_isProxy) {
    std::vector<com::PtrRequest> currentRecvRequests;
    for (const auto &task : _proxyRecvTasks) {
      double               *recvBuf = reinterpret_cast<double *>(_recvBasePtr + task.shmOffset);
      precice::span<double> aggregatedSpan(recvBuf, task.totalDoubles);
      currentRecvRequests.push_back(_communication->aReceive(aggregatedSpan, task.targetRank));
    }
    for (auto &req : currentRecvRequests)
      req->wait();
  }

  MPI_Barrier(_localComm);

  for (const auto &task : _workerRecvTasks) {
    for (size_t i = 0; i < task.count; ++i) {
      int vertexIndex = task.indicesPtr[i];
      for (int d = 0; d < valueDimension; ++d) {
        itemsToReceive[vertexIndex * valueDimension + d] += task.shmPtr[i * valueDimension + d];
      }
    }
  }
}

// =========================================================================
// [修复 5] 终极版辅助通信函数：路由代理 + 去重 + 本地同步
// =========================================================================

void HierarchicalCommunication::broadcastSend(int itemToSend)
{
  if (!_isProxy || !_communication)
    return;

  std::set<int> sentProxies;
  for (auto &connectionData : _connectionDataVector) {
    if (_remoteRankToProxy.count(connectionData.remoteRank)) {
      int targetProxy = _remoteRankToProxy[connectionData.remoteRank];
      if (sentProxies.find(targetProxy) == sentProxies.end()) {
        _communication->send(itemToSend, targetProxy);
        sentProxies.insert(targetProxy);
      }
    }
  }
}

void HierarchicalCommunication::broadcastReceiveAll(std::vector<int> &itemToReceive)
{
  std::map<int, int> proxyData;

  if (_isProxy && _communication) {
    std::set<int> recvProxies;
    for (auto &connectionData : _connectionDataVector) {
      if (_remoteRankToProxy.count(connectionData.remoteRank)) {
        int targetProxy = _remoteRankToProxy[connectionData.remoteRank];
        if (recvProxies.find(targetProxy) == recvProxies.end()) {
          int val = 0;
          _communication->receive(val, targetProxy);
          proxyData[targetProxy] = val;
          recvProxies.insert(targetProxy);
        }
      }
    }
  }

  for (auto &connectionData : _connectionDataVector) {
    int data = 0;
    if (_isProxy && _remoteRankToProxy.count(connectionData.remoteRank)) {
      data = proxyData[_remoteRankToProxy[connectionData.remoteRank]];
    }
    MPI_Bcast(&data, 1, MPI_INT, 0, _localComm);
    itemToReceive.push_back(data);
  }
}

void HierarchicalCommunication::broadcastSendMesh()
{
  if (!_isProxy || !_communication)
    return;

  std::set<int> sentProxies;
  for (auto &connectionData : _connectionDataVector) {
    if (_remoteRankToProxy.count(connectionData.remoteRank)) {
      int targetProxy = _remoteRankToProxy[connectionData.remoteRank];
      if (sentProxies.find(targetProxy) == sentProxies.end()) {
        com::sendMesh(*_communication, targetProxy, *_mesh);
        sentProxies.insert(targetProxy);
      }
    }
  }
}

void HierarchicalCommunication::broadcastReceiveAllMesh()
{
  if (_isProxy && _communication) {
    std::set<int> recvProxies;
    for (auto &connectionData : _connectionDataVector) {
      if (_remoteRankToProxy.count(connectionData.remoteRank)) {
        int targetProxy = _remoteRankToProxy[connectionData.remoteRank];
        if (recvProxies.find(targetProxy) == recvProxies.end()) {
          com::receiveMesh(*_communication, targetProxy, *_mesh);
          recvProxies.insert(targetProxy);
        }
      }
    }
  }
}

void HierarchicalCommunication::scatterAllCommunicationMap(CommunicationMap &localCommunicationMap)
{
  if (!_isProxy || !_communication)
    return;

  for (auto &connectionData : _connectionDataVector) {
    if (_remoteRankToProxy.count(connectionData.remoteRank)) {
      int targetProxy = _remoteRankToProxy[connectionData.remoteRank];
      _communication->sendRange(localCommunicationMap[connectionData.remoteRank], targetProxy);
    }
  }
}

void HierarchicalCommunication::gatherAllCommunicationMap(CommunicationMap &localCommunicationMap)
{
  for (auto &connectionData : _connectionDataVector) {
    std::vector<int> recvVec;
    int              vecSize = 0;

    if (_isProxy && _communication) {
      if (_remoteRankToProxy.count(connectionData.remoteRank)) {
        int targetProxy = _remoteRankToProxy[connectionData.remoteRank];
        recvVec         = _communication->receiveRange(targetProxy, com::asVector<int>);
        vecSize         = recvVec.size();
      }
    }

    MPI_Bcast(&vecSize, 1, MPI_INT, 0, _localComm);

    if (!_isProxy)
      recvVec.resize(vecSize);

    if (vecSize > 0)
      MPI_Bcast(recvVec.data(), vecSize, MPI_INT, 0, _localComm);

    localCommunicationMap[connectionData.remoteRank] = recvVec;
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
      std::this_thread::yield();
  } while (blocking);
}

void HierarchicalCommunication::waitAllOngoingRequests()
{
  if (!_ongoingRequests.empty()) {
    for (auto &req : _ongoingRequests)
      req->wait();
    _ongoingRequests.clear();
  }
}

void HierarchicalCommunication::initializeSendPattern(int valueDimension)
{
  freeSendWindow();

  MPI_Comm localComm    = _localComm;
  int      localSize    = _localSize;
  int      myGlobalRank = utils::IntraComm::getRank();

  std::vector<long> mySendMeta;
  for (const auto &mapping : _mappings) {
    if (_remoteRankToProxy.find(mapping.remoteRank) == _remoteRankToProxy.end())
      continue;

    long bytes = static_cast<long>(mapping.indices.size() * valueDimension * sizeof(double));
    if (bytes > 0) {
      mySendMeta.push_back(_remoteRankToProxy[mapping.remoteRank]);
      mySendMeta.push_back(mapping.remoteRank);
      mySendMeta.push_back(myGlobalRank);
      mySendMeta.push_back(bytes);
    }
  }

  std::vector<int> counts(localSize);
  int              mySize = static_cast<int>(mySendMeta.size());
  MPI_Allgather(&mySize, 1, MPI_INT, counts.data(), 1, MPI_INT, localComm);

  std::vector<int> displs(localSize + 1, 0);
  for (int i = 0; i < localSize; ++i)
    displs[i + 1] = displs[i] + counts[i];

  std::vector<long> globalMeta(displs.back());

  MPI_Allgatherv(mySendMeta.data(), mySize, MPI_LONG,
                 globalMeta.data(), counts.data(), displs.data(), MPI_LONG,
                 localComm);

  struct ReqEntry {
    long target, remote, source, bytes, offset;
  };
  std::vector<ReqEntry> requests;
  requests.reserve(globalMeta.size() / 4);

  for (size_t i = 0; i < globalMeta.size(); i += 4) {
    requests.push_back({globalMeta[i], globalMeta[i + 1], globalMeta[i + 2], globalMeta[i + 3], 0});
  }

  std::sort(requests.begin(), requests.end(), [](const ReqEntry &a, const ReqEntry &b) {
    if (a.target != b.target)
      return a.target < b.target;
    if (a.remote != b.remote)
      return a.remote < b.remote;
    return a.source < b.source;
  });

  long currentOffset = 0;

  for (auto &req : requests) {
    req.offset = currentOffset;
    if (_proxySendTasks.empty() || _proxySendTasks.back().targetRank != req.target) {
      _proxySendTasks.push_back({static_cast<int>(req.target), currentOffset, 0});
    }
    _proxySendTasks.back().totalDoubles += req.bytes / sizeof(double);
    currentOffset += req.bytes;
  }

  long totalSize = (currentOffset + 7) & ~7;

  MPI_Info winInfo;
  MPI_Info_create(&winInfo);
  MPI_Info_set(winInfo, "alloc_shared_noncontig", "true");

  MPI_Aint size = _isProxy ? totalSize : 0;
  int      ret  = MPI_Win_allocate_shared(size, sizeof(char), winInfo, localComm, &_sendBasePtr, &_winSend);
  PRECICE_ASSERT(ret == MPI_SUCCESS, "MPI_Win_allocate_shared failed for Send Window");
  MPI_Info_free(&winInfo);

  if (!_isProxy) {
    MPI_Aint sz;
    int      dsp;
    MPI_Win_shared_query(_winSend, 0, &sz, &dsp, &_sendBasePtr);
  }

  for (const auto &req : requests) {
    if (req.source == myGlobalRank) {
      for (const auto &m : _mappings) {
        if (static_cast<long>(m.remoteRank) == req.remote) {
          _workerSendTasks.push_back({reinterpret_cast<double *>(_sendBasePtr + req.offset),
                                      m.indices.data(),
                                      m.indices.size()});
          break;
        }
      }
    }
  }

  _cachedSendDim = valueDimension;
}

void HierarchicalCommunication::initializeRecvPattern(int valueDimension)
{
  freeRecvWindow();

  MPI_Comm localComm    = _localComm;
  int      localSize    = _localSize;
  int      myGlobalRank = utils::IntraComm::getRank();

  std::vector<long> myRecvMeta;
  for (const auto &mapping : _mappings) {
    if (_remoteRankToProxy.find(mapping.remoteRank) == _remoteRankToProxy.end())
      continue;

    long bytes = static_cast<long>(mapping.indices.size() * valueDimension * sizeof(double));
    if (bytes > 0) {
      myRecvMeta.push_back(_remoteRankToProxy[mapping.remoteRank]);
      myRecvMeta.push_back(myGlobalRank);
      myRecvMeta.push_back(mapping.remoteRank);
      myRecvMeta.push_back(bytes);
    }
  }

  std::vector<int> counts(localSize);
  int              mySize = static_cast<int>(myRecvMeta.size());
  MPI_Allgather(&mySize, 1, MPI_INT, counts.data(), 1, MPI_INT, localComm);

  std::vector<int> displs(localSize + 1, 0);
  for (int i = 0; i < localSize; ++i)
    displs[i + 1] = displs[i] + counts[i];

  std::vector<long> globalMeta(displs.back());

  MPI_Allgatherv(myRecvMeta.data(), mySize, MPI_LONG,
                 globalMeta.data(), counts.data(), displs.data(), MPI_LONG,
                 localComm);

  struct RecvReqEntry {
    long remoteProxy, myRank, remoteRank, bytes, offset;
  };

  std::vector<RecvReqEntry> requests;
  requests.reserve(globalMeta.size() / 4);
  for (size_t i = 0; i < globalMeta.size(); i += 4) {
    requests.push_back({globalMeta[i], globalMeta[i + 1], globalMeta[i + 2], globalMeta[i + 3], 0});
  }

  std::sort(requests.begin(), requests.end(),
            [](const RecvReqEntry &a, const RecvReqEntry &b) {
              if (a.remoteProxy != b.remoteProxy)
                return a.remoteProxy < b.remoteProxy;
              if (a.myRank != b.myRank)
                return a.myRank < b.myRank;
              return a.remoteRank < b.remoteRank;
            });

  long currentOffset = 0;

  for (auto &req : requests) {
    req.offset = currentOffset;
    if (_proxyRecvTasks.empty() || _proxyRecvTasks.back().targetRank != req.remoteProxy) {
      _proxyRecvTasks.push_back({static_cast<int>(req.remoteProxy), currentOffset, 0});
    }
    _proxyRecvTasks.back().totalDoubles += req.bytes / sizeof(double);
    currentOffset += req.bytes;
  }

  long totalSize = (currentOffset + 7) & ~7;

  MPI_Info winInfo;
  MPI_Info_create(&winInfo);
  MPI_Info_set(winInfo, "alloc_shared_noncontig", "true");

  MPI_Aint size = _isProxy ? totalSize : 0;
  int      ret  = MPI_Win_allocate_shared(size, sizeof(char), winInfo, localComm, &_recvBasePtr, &_winRecv);
  PRECICE_ASSERT(ret == MPI_SUCCESS, "MPI_Win_allocate_shared failed for Recv Window");
  MPI_Info_free(&winInfo);

  if (!_isProxy) {
    MPI_Aint sz;
    int      dsp;
    MPI_Win_shared_query(_winRecv, 0, &sz, &dsp, &_recvBasePtr);
  }

  for (const auto &req : requests) {
    if (req.myRank == myGlobalRank) {
      for (const auto &m : _mappings) {
        if (static_cast<long>(m.remoteRank) == req.remoteRank) {
          _workerRecvTasks.push_back({reinterpret_cast<double *>(_recvBasePtr + req.offset),
                                      m.indices.data(),
                                      m.indices.size()});
          break;
        }
      }
    }
  }

  _cachedRecvDim = valueDimension;
}

void HierarchicalCommunication::freeSendWindow()
{
  if (_winSend != MPI_WIN_NULL) {
    MPI_Win_free(&_winSend);
    _winSend     = MPI_WIN_NULL;
    _sendBasePtr = nullptr;
  }
  _proxySendTasks.clear();
  _workerSendTasks.clear();
}

void HierarchicalCommunication::freeRecvWindow()
{
  if (_winRecv != MPI_WIN_NULL) {
    MPI_Win_free(&_winRecv);
    _winRecv     = MPI_WIN_NULL;
    _recvBasePtr = nullptr;
  }
  _proxyRecvTasks.clear();
  _workerRecvTasks.clear();
}
} // namespace precice::m2n