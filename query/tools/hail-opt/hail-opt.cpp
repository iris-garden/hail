#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "InitAllDialects.h"
#include "InitAllPasses.h"

int main(int argc, char **argv) {
  // mlir::registerAllPasses();

  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  mlir::registerConvertAffineToStandardPass();
  mlir::registerConvertLinalgToStandardPass();
  mlir::registerConvertTensorToLinalgPass();
  mlir::registerConvertVectorToSCFPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerSCFToControlFlowPass();

  // Dialect passes
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerLinalgLowerToAffineLoopsPass();
  mlir::registerLinalgLowerToLoopsPass();

  // Hail passes
  hail::ir::registerAllPasses();

  // Dialects
  mlir::DialectRegistry registry;
  hail::ir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Hail optimizer driver\n", registry));
}
