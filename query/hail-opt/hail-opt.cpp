#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "InitAllPasses.h"

// dialect includes
#include "Dialect/Sandbox/IR/Sandbox.h"

int main(int argc, char **argv) {
  // mlir::registerAllPasses();
  mlir::registerCanonicalizerPass();
  hail::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithmeticDialect, mlir::scf::SCFDialect,
                  mlir::func::FuncDialect, hail::ir::SandboxDialect>();

  // Register additional dialects below. Only dialects that will be *parsed*
  // by the tool need be registered, not the ones generated by any passes the
  // tool runs.

  // Uncomment the line below to include a few standard dialects.
  // registry.insert<mlir::scf::SCFDialect, mlir::LLVM::LLVMDialect>();
  // Uncomment the line below to include *all* MLIR Core dialects
  // registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Hail optimizer driver\n", registry));
}
