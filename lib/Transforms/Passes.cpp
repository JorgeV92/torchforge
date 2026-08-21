//===- Passes.cpp - TorchForge custom MLIR passes -------------------------===//
#include "TorchForge/Passes.h"
#include "TorchForge/Dialect/TorchForgeOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace torchforge {
namespace {

static bool isIntegerConstant(mlir::Value value, int64_t expected) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto attr = llvm::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return attr && attr.getValue().getSExtValue() == expected;
}

static bool isFloatOne(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto attr = llvm::dyn_cast<mlir::FloatAttr>(constant.getValue());
  return attr && attr.getValue().isExactlyValue(1.0);
}

struct FoldAddIZero : mlir::OpRewritePattern<mlir::arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::AddIOp op,
      mlir::PatternRewriter &rewriter) const override {
    if (isIntegerConstant(op.getLhs(), 0)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }
    if (isIntegerConstant(op.getRhs(), 0)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct FoldSubIZero : mlir::OpRewritePattern<mlir::arith::SubIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::SubIOp op,
      mlir::PatternRewriter &rewriter) const override {
    if (!isIntegerConstant(op.getRhs(), 0))
      return mlir::failure();
    rewriter.replaceOp(op, op.getLhs());
    return mlir::success();
  }
};

struct FoldMulIOne : mlir::OpRewritePattern<mlir::arith::MulIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::MulIOp op,
      mlir::PatternRewriter &rewriter) const override {
    if (isIntegerConstant(op.getLhs(), 1)) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }
    if (isIntegerConstant(op.getRhs(), 1)) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct FoldMulFOne : mlir::OpRewritePattern<mlir::arith::MulFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::MulFOp op,
      mlir::PatternRewriter &rewriter) const override {
    if (isFloatOne(op.getLhs())) {
      rewriter.replaceOp(op, op.getRhs());
      return mlir::success();
    }
    if (isFloatOne(op.getRhs())) {
      rewriter.replaceOp(op, op.getLhs());
      return mlir::success();
    }
    return mlir::failure();
  }
};

} // namespace 

} // namespace torchforge
