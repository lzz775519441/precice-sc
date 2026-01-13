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
  waitAllOngoingRequests();
  freeSendWindow();
  freeRecvWindow();
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

  waitAllOngoingRequests();
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

  if (_isProxy) {
    waitAllOngoingRequests();
  }
  mpiCom->sharedMemoryBarrier();

  if (valueDimension != _cachedSendDim) {
      initializeSendPattern(valueDimension);
  }

  // Worker 并行写入
  for (const auto& task : _workerSendTasks) {
    for (size_t i = 0; i < task.count; ++i) {
      int vertexIndex = task.indicesPtr[i];
      for (int d = 0; d < valueDimension; ++d) {
        // Copy User Data -> Shared Memory Window
        task.shmPtr[i * valueDimension + d] = itemsToSend[vertexIndex * valueDimension + d];
      }
    }
  }

  mpiCom->sharedMemoryBarrier();

  // Proxy 聚合发送
  if (_isProxy) {
    for (const auto& task : _proxySendTasks) {
      double* sendBuf = reinterpret_cast<double*>(_sendBasePtr + task.shmOffset);
      precice::span<double const> aggregatedSpan(sendBuf, task.totalDoubles);
      _ongoingRequests.push_back(_communication->aSend(aggregatedSpan, task.targetRank));
    }
  }

  mpiCom->sharedMemoryBarrier();
}

void HierarchicalCommunication::receive(precice::span<double> itemsToReceive, int valueDimension)
{

  std::fill(itemsToReceive.begin(), itemsToReceive.end(), 0.0);

  auto mpiCom = std::dynamic_pointer_cast<com::MPICommunication>(_communication);
  PRECICE_ASSERT(mpiCom, "Hierarchical communication requires MPI.");

  if (_isProxy) {
    waitAllOngoingRequests();
  }
  mpiCom->sharedMemoryBarrier();

  if (valueDimension != _cachedRecvDim) {
      initializeRecvPattern(valueDimension);
  }

  // Proxy 聚合接收
  if (_isProxy) {
    std::vector<com::PtrRequest> currentRecvRequests;

    for (const auto& task : _proxyRecvTasks) {
      double* recvBuf = reinterpret_cast<double*>(_recvBasePtr + task.shmOffset);
      precice::span<double> aggregatedSpan(recvBuf, task.totalDoubles);
      currentRecvRequests.push_back(_communication->aReceive(aggregatedSpan, task.targetRank));
    }
    for (auto& req : currentRecvRequests) {
        req->wait();
    }
  }

  mpiCom->sharedMemoryBarrier();

  // Worker 并行读取
  for (const auto& task : _workerRecvTasks) {
    for (size_t i = 0; i < task.count; ++i) {
      int vertexIndex = task.indicesPtr[i];
      for (int d = 0; d < valueDimension; ++d) {
        itemsToReceive[vertexIndex * valueDimension + d] += task.shmPtr[i * valueDimension + d];
      }
    }
  }

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
      req->wait();
    }
    _ongoingRequests.clear();
  }
}

void HierarchicalCommunication::initializeSendPattern(int valueDimension)
{
  // 1. 清理旧资源 (防止内存泄漏或窗口重叠)
  freeSendWindow();

  auto mpiCom = std::dynamic_pointer_cast<com::MPICommunication>(_communication);
  PRECICE_ASSERT(mpiCom, "Hierarchical communication requires MPI.");

  // 2. 确保拓扑信息已建立
  if (_remoteRankToProxy.empty()) {
    exchangeTopology();
  }

  MPI_Comm localComm = mpiCom->getLocalCommunicator();
  int localSize = mpiCom->getLocalSize();
  int myGlobalRank = utils::IntraComm::getRank();


  // ===========================================================================
  // 步骤一：构建与交换元数据 (Metadata)
  // ===========================================================================

  // 1.1 收集本地发送需求
  // 格式: [TargetProxy, RemoteRank, SourceRank, Bytes]
  std::vector<long> mySendMeta;
  for (const auto &mapping : _mappings) {
    // 过滤掉未知的远程 Rank
    if (_remoteRankToProxy.find(mapping.remoteRank) == _remoteRankToProxy.end()) {
        continue;
    }

    long bytes = static_cast<long>(mapping.indices.size() * valueDimension * sizeof(double));
    if (bytes > 0) {
      mySendMeta.push_back(_remoteRankToProxy[mapping.remoteRank]); // TargetProxy (聚合目标)
      mySendMeta.push_back(mapping.remoteRank);                     // RemoteRank (最终目的地)
      mySendMeta.push_back(myGlobalRank);                           // SourceRank (我)
      mySendMeta.push_back(bytes);
    }
  }

  // 1.2 节点内全收集 (Allgatherv)
  std::vector<int> counts(localSize);
  int mySize = static_cast<int>(mySendMeta.size());
  MPI_Allgather(&mySize, 1, MPI_INT, counts.data(), 1, MPI_INT, localComm);

  std::vector<int> displs(localSize + 1, 0);
  for (int i = 0; i < localSize; ++i) {
    displs[i + 1] = displs[i] + counts[i];
  }
  std::vector<long> globalMeta(displs.back());

  MPI_Allgatherv(mySendMeta.data(), mySize, MPI_LONG,
                 globalMeta.data(), counts.data(), displs.data(), MPI_LONG,
                 localComm);

  // ===========================================================================
  // 步骤二：排序与布局计算
  // ===========================================================================

  // 2.1 解析为结构体以便排序
  struct ReqEntry {
      long target;
      long remote;
      long source;
      long bytes;
      long offset;
  };
  std::vector<ReqEntry> requests;
  requests.reserve(globalMeta.size() / 4);

  for (size_t i = 0; i < globalMeta.size(); i += 4) {
    requests.push_back({globalMeta[i], globalMeta[i+1], globalMeta[i+2], globalMeta[i+3], 0});
  }

  // 2.2 确定性排序 (关键)
  // 必须保证所有进程算出的 offset 顺序一致
  // 优先级: TargetProxy -> RemoteRank -> SourceRank
  std::sort(requests.begin(), requests.end(), [](const ReqEntry &a, const ReqEntry &b) {
      if (a.target != b.target) return a.target < b.target;
      if (a.remote != b.remote) return a.remote < b.remote;
      return a.source < b.source;
  });

  // 2.3 计算内存偏移量 & 生成 Proxy 任务
  long currentOffset = sizeof(SharedMemoryHeader); // 预留 Header 空间

  for (auto &req : requests) {
    req.offset = currentOffset;

    // 如果我是 Proxy，记录聚合发送任务
    // 逻辑：只要 Target 变了，就是新的一批聚合包
    if (_proxySendTasks.empty() || _proxySendTasks.back().targetRank != req.target) {
        _proxySendTasks.push_back({static_cast<int>(req.target), currentOffset, 0});
    }
    _proxySendTasks.back().totalDoubles += req.bytes / sizeof(double);

    currentOffset += req.bytes;
  }

  // 内存对齐 (8字节对齐)，虽然这里不是必须的，但为了性能是个好习惯
  long totalSize = (currentOffset + 7) & ~7;

  // ===========================================================================
  // 步骤三：申请独立的共享窗口 (Send Window)
  // ===========================================================================

  MPI_Info winInfo;
  MPI_Info_create(&winInfo);
  MPI_Info_set(winInfo, "alloc_shared_noncontig", "true"); // 允许非连续内存优化

  // 只有 Proxy 申请实际物理内存，Worker 申请大小为 0
  MPI_Aint size = _isProxy ? totalSize : 0;
  int ret = MPI_Win_allocate_shared(size, sizeof(char), winInfo, localComm, &_sendBasePtr, &_winSend);
  PRECICE_ASSERT(ret == MPI_SUCCESS, "MPI_Win_allocate_shared failed for Send Window");
  MPI_Info_free(&winInfo);

  // Worker 需要查询 Proxy (Rank 0) 的地址
  if (!_isProxy) {
      MPI_Aint sz;
      int dsp;
      // 查询 Rank 0 在 _winSend 上的地址
      MPI_Win_shared_query(_winSend, 0, &sz, &dsp, &_sendBasePtr);
  }

  // 3.1 初始化 Header (仅 Proxy)
  if (_isProxy && _sendBasePtr) {
      SharedMemoryHeader* header = reinterpret_cast<SharedMemoryHeader*>(_sendBasePtr);
      header->payloadSize = totalSize - sizeof(SharedMemoryHeader);
      header->stateFlag = 0; // 初始化状态
  }
  // 确保内存分配完成且 Header 可见
  MPI_Win_fence(0, _winSend);

  // ===========================================================================
  // 步骤四：缓存 Worker 任务 (预计算指针)
  // ===========================================================================

  for (const auto &req : requests) {
    // 如果这条请求是我发起的 (Source == Me)
    if (req.source == myGlobalRank) {
      // 查找对应的 mapping 以获取 indices
      for (const auto &m : _mappings) {
          if (static_cast<long>(m.remoteRank) == req.remote) {
              _workerSendTasks.push_back({
                  reinterpret_cast<double*>(_sendBasePtr + req.offset), // 绝对物理地址
                  m.indices.data(),
                  m.indices.size()
              });
              break;
          }
      }
    }
  }

  _cachedSendDim = valueDimension;
  MPI_Barrier(localComm);
}

void HierarchicalCommunication::initializeRecvPattern(int valueDimension)
{
  // 1. 清理旧资源 (关键：防止窗口重叠或泄漏)
  freeRecvWindow();

  auto mpiCom = std::dynamic_pointer_cast<com::MPICommunication>(_communication);
  PRECICE_ASSERT(mpiCom, "Hierarchical communication requires MPI.");

  // 2. 确保拓扑信息已建立
  if (_remoteRankToProxy.empty()) {
    exchangeTopology();
  }

  MPI_Comm localComm = mpiCom->getLocalCommunicator();
  int localSize = mpiCom->getLocalSize();
  int myGlobalRank = utils::IntraComm::getRank();


  // ===========================================================================
  // 步骤一：构建与交换元数据 (Metadata)
  // ===========================================================================

  // 1.1 收集本地接收需求
  // 格式: [RemoteProxy, MyRank(Receiver), RemoteRank(Sender), Bytes]
  // 注意：这里的 RemoteProxy 是对方的 Proxy（即数据的来源聚合点）
  std::vector<long> myRecvMeta;
  for (const auto &mapping : _mappings) {
    if (_remoteRankToProxy.find(mapping.remoteRank) == _remoteRankToProxy.end()) {
        continue;
    }
    long bytes = static_cast<long>(mapping.indices.size() * valueDimension * sizeof(double));
    if (bytes > 0) {
      myRecvMeta.push_back(_remoteRankToProxy[mapping.remoteRank]); // RemoteProxy (Sender Proxy)
      myRecvMeta.push_back(myGlobalRank);                           // MyRank (Receiver)
      myRecvMeta.push_back(mapping.remoteRank);                     // RemoteRank (Sender)
      myRecvMeta.push_back(bytes);
    }
  }

  // 1.2 节点内全收集 (Allgatherv)
  std::vector<int> counts(localSize);
  int mySize = static_cast<int>(myRecvMeta.size());
  MPI_Allgather(&mySize, 1, MPI_INT, counts.data(), 1, MPI_INT, localComm);

  std::vector<int> displs(localSize + 1, 0);
  for (int i = 0; i < localSize; ++i) {
    displs[i + 1] = displs[i] + counts[i];
  }
  std::vector<long> globalMeta(displs.back());

  MPI_Allgatherv(myRecvMeta.data(), mySize, MPI_LONG,
                 globalMeta.data(), counts.data(), displs.data(), MPI_LONG,
                 localComm);

  // ===========================================================================
  // 步骤二：排序与布局计算
  // ===========================================================================

  // 2.1 解析为结构体
  struct RecvReqEntry {
    long remoteProxy;
    long myRank;
    long remoteRank;
    long bytes;
    long offset;
  };

  std::vector<RecvReqEntry> requests;
  requests.reserve(globalMeta.size() / 4);
  for (size_t i = 0; i < globalMeta.size(); i += 4) {
    requests.push_back({globalMeta[i], globalMeta[i+1], globalMeta[i+2], globalMeta[i+3], 0});
  }

  // 2.2 确定性排序 (关键)
  // 排序必须与发送端 (SendPattern) 的逻辑对应，以确保逻辑清晰
  // 优先级: RemoteProxy -> MyRank (Receiver) -> RemoteRank (Sender)
  std::sort(requests.begin(), requests.end(),
    [](const RecvReqEntry &a, const RecvReqEntry &b) {
      if (a.remoteProxy != b.remoteProxy) return a.remoteProxy < b.remoteProxy;
      if (a.myRank != b.myRank)           return a.myRank < b.myRank;
      return a.remoteRank < b.remoteRank;
    });

  // 2.3 计算内存偏移量 & 生成 Proxy 任务
  long currentOffset = sizeof(SharedMemoryHeader);

  for (auto &req : requests) {
    req.offset = currentOffset;

    // 如果我是 Proxy，记录聚合接收任务
    // 逻辑：只要 RemoteProxy 变了，就是来自不同节点的一批新数据
    if (_proxyRecvTasks.empty() || _proxyRecvTasks.back().targetRank != req.remoteProxy) {
        _proxyRecvTasks.push_back({static_cast<int>(req.remoteProxy), currentOffset, 0});
    }
    _proxyRecvTasks.back().totalDoubles += req.bytes / sizeof(double);

    currentOffset += req.bytes;
  }

  // 内存对齐
  long totalSize = (currentOffset + 7) & ~7;

  // ===========================================================================
  // 步骤三：申请独立的共享窗口 (Recv Window)
  // ===========================================================================

  MPI_Info winInfo;
  MPI_Info_create(&winInfo);
  MPI_Info_set(winInfo, "alloc_shared_noncontig", "true");

  // 只有 Proxy 申请实际大小
  MPI_Aint size = _isProxy ? totalSize : 0;
  // [关键] 申请 _winRecv 并获取本地指针 _recvBasePtr
  int ret = MPI_Win_allocate_shared(size, sizeof(char), winInfo, localComm, &_recvBasePtr, &_winRecv);
  PRECICE_ASSERT(ret == MPI_SUCCESS, "MPI_Win_allocate_shared failed for Recv Window");
  MPI_Info_free(&winInfo);

  // Worker 查询地址
  if (!_isProxy) {
      MPI_Aint sz;
      int dsp;
      MPI_Win_shared_query(_winRecv, 0, &sz, &dsp, &_recvBasePtr);
  }

  // 初始化 Header (仅 Proxy)
  if (_isProxy && _recvBasePtr) {
      SharedMemoryHeader* header = reinterpret_cast<SharedMemoryHeader*>(_recvBasePtr);
      header->payloadSize = totalSize - sizeof(SharedMemoryHeader);
      header->stateFlag = 0;
  }
  // 确保内存分配完成
  MPI_Win_fence(0, _winRecv);

  // ===========================================================================
  // 步骤四：缓存 Worker 任务 (预计算指针)
  // ===========================================================================

  for (const auto &req : requests) {
    // 如果这条请求是发给我的 (MyRank == Me)
    if (req.myRank == myGlobalRank) {
      // 查找对应的 mapping
      for (const auto &m : _mappings) {
          if (static_cast<long>(m.remoteRank) == req.remoteRank) {
              _workerRecvTasks.push_back({
                  reinterpret_cast<double*>(_recvBasePtr + req.offset), // 绝对物理地址
                  m.indices.data(),
                  m.indices.size()
              });
              break;
          }
      }
    }
  }

  _cachedRecvDim = valueDimension;
  MPI_Barrier(localComm);
}

void HierarchicalCommunication::freeSendWindow() {
  if (_winSend != MPI_WIN_NULL) {
    MPI_Win_free(&_winSend);
    _winSend = MPI_WIN_NULL;
    _sendBasePtr = nullptr;
  }
  _proxySendTasks.clear();
  _workerSendTasks.clear();
}

void HierarchicalCommunication::freeRecvWindow() {
  if (_winRecv != MPI_WIN_NULL) {
    MPI_Win_free(&_winRecv);
    _winRecv = MPI_WIN_NULL;
    _recvBasePtr = nullptr;
  }
  _proxyRecvTasks.clear();
  _workerRecvTasks.clear();
}
} // namespace precice::m2n
