#include "HierarchicalComFactory.hpp"

#include "HierarchicalCommunication.hpp"

#include <utility>

#include "com/SharedPointer.hpp"

namespace precice::m2n {

HierarchicalComFactory::HierarchicalComFactory(com::PtrCommunicationFactory comFactory)
    : _comFactory(std::move(comFactory)) {}

DistributedCommunication::SharedPointer
HierarchicalComFactory::newDistributedCommunication(mesh::PtrMesh mesh)
{
  return DistributedCommunication::SharedPointer(new HierarchicalCommunication(_comFactory, mesh));
}


} // namespace precice::m2n