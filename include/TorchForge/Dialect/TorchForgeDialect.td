#ifndef TORCHFORGE_DIALECT_TORCHFORGEDIALECT_TD
#define TORCHFORGE_DIALECT_TORCHFORGEDIALECT_TD

include "mlir/IR/OpBase.td"

def TorchForge_Dialect : Dialect {
  let name = "torchforge";
  let cppNamespace = "::torchforge";
  let summary = "Educational ML compiler operations used by TorchForge";
  let description = [{
    The TorchForge dialect contains small operations that make compiler concepts
    visible while models move from PyTorch into progressively lower-level MLIR.
  }];
}

class TorchForge_Op<string mnemonic, list<Trait> traits = []>
    : Op<TorchForge_Dialect, mnemonic, traits>;

#endif
