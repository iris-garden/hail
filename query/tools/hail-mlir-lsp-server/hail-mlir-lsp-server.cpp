//===- mlir-lsp-server.cpp - MLIR Language Server -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

#include "Dialect/CPS/IR/CPS.h"
#include "Dialect/Missing/IR/Missing.h"
#include "Dialect/Sandbox/IR/Sandbox.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  // registerAllDialects(registry);
  registry.insert<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect, hail::ir::CPSDialect,
                  hail::ir::MissingDialect, hail::ir::SandboxDialect>();

  return failed(MlirLspServerMain(argc, argv, registry));
}
