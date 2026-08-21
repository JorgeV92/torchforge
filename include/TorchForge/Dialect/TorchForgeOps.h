#ifndef TORCHFORGE_DIALECT_TORCHFORGEOPS_H
#define TORCHFORGE_DIALECT_TORCHFORGEOPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TorchForge/Dialect/TorchForgeOps.h.inc"

#endif
