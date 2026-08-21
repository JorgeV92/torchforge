//===- Passes.h - TorchForge custom MLIR passes -----------------*- C++ -*-===//
#ifndef TORCHFORGE_PASSES_H
#define TORCHFORGE_PASSES_H

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
} // namespace mlir

namespace torchforge {

std::unique_ptr<mlir::Pass> createOpStatsPass();
std::unique_ptr<mlir::Pass> createAnnotateCostPass();
std::unique_ptr<mlir::Pass> createFoldArithIdentitiesPass();
std::unique_ptr<mlir::Pass> createMarkExpensiveFunctionsPass();
std::unique_ptr<mlir::Pass> createLowerObservePass();
std::unique_ptr<mlir::Pass> createLinalgReportPass();

void buildTorchForgeCleanupPipeline(mlir::OpPassManager &pm);
void registerTorchForgePasses();

} // namespace torchforge

#endif // TORCHFORGE_PASSES_H
