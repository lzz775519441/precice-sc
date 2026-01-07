#include "PointToPointComFactory.hpp"

#include <utility>

#include "PointToPointCommunication.hpp"

#include "com/SharedPointer.hpp"

namespace precice::m2n {

PointToPointComFactory::PointToPointComFactory(com::PtrCommunicationFactory comFactory)
    : _comFactory(std::move(comFactory)) {}

DistributedCommunication::SharedPointer
PointToPointComFactory::newDistributedCommunication(mesh::PtrMesh mesh)
{
  return DistributedCommunication::SharedPointer(new PointToPointCommunication(_comFactory, mesh));
}

com::PtrCommunicationFactory PointToPointComFactory::communicationFactory()
{
  return _comFactory;
}

} // namespace precice::m2n
