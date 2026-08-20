# TorchForge 

TorchForge is an experimental machine-learning compiler project for exploring how PyTorch programs are captured, represented, transformed, and lowered using Torch-MLIR, MLIR, and C++.

The project is designed as a learning and experimentation environment for ML compiler development. It connects a familiar PyTorch frontend with a native MLIR compiler pipeline so that compiler transformations can be studied and implemented directly in C++.

TorchForge is intended to be extended over time with new MLIR passes, dialect operations, analyses, lowering pipelines, and compiler backends.