#ifndef DIALECT_CPS_IR_CPS_H
#define DIALECT_CPS_IR_CPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/IR/BuiltinTypes.h"

#include "Dialect/CPS/IR/CPSOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/CPS/IR/CPSOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Dialect/CPS/IR/CPSOps.h.inc"

#endif // DIALECT_CPS_IR_CPS_H
