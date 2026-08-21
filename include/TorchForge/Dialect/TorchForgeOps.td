#ifndef TORCHFORGE_DIALECT_TORCHFORGEOPS_TD
#define TORCHFORGE_DIALECT_TORCHFORGEOPS_TD

include "TorchForge/Dialect/TorchForgeDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TorchForge_ObserveOp
    : TorchForge_Op<"observe", [Pure, SameOperandsAndResultType]> {
  let summary = "Attach an educational observation point to a value";
  let description = [{
    `torchforge.observe` is semantically an identity operation. It carries a
    string tag so compiler students can insert named observation points into IR,
    inspect them between passes, and then lower them away.

    Example:

    ```mlir
    %0 = torchforge.observe %arg0 tag = "model_input" : tensor<4xf32>
    ```
  }];

  let arguments = (ins AnyType:$input, StrAttr:$tag);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input `tag` `=` $tag attr-dict `:` type($input)";
}

#endif
