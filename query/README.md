# [MLIR](https://mlir.llvm.org) + [hail](https://hail.is) = 🚀🧬?

## Setting up your shell environment

In your shell configuration file (e.g. `~/.bashrc` for `bash`), add the
following:

```sh
export HAIL_DIRECTORY=$HOME/src/hail
export HAIL_NATIVE_COMPILER_DIRECTORY=$HAIL_DIRECTORY/query
export HAIL_NATIVE_COMPILER_BUILD_DIRECTORY=$HAIL_NATIVE_COMPILER_DIRECTORY/build

export LLVM_DIRECTORY=$HOME/src/llvm-project
export LLVM_BUILD_DIRECTORY=$LLVM_DIRECTORY/build
export LLVM_BUILD_BIN_DIRECTORY=$LLVM_BUILD_DIRECTORY/bin

export PATH=$LLVM_BUILD_BIN_DIRECTORY:$PATH
```

Replace the path for `HAIL_DIRECTORY` with the path to the root of your local
clone of the Hail repository.

Replace the path for `LLVM_DIRECTORY` with the path where you would like the
local clone of the LLVM repository to live (but make sure to include
`llvm-project` at the end, as this is the name of the folder that will be the
root of the repository).

Add the following to your shell configuration file to include the build scripts
on your `PATH` as well, if they're not already on there:

```sh
export HAIL_DEVBIN_DIRECTORY=$HAIL_DIRECTORY/devbin
export PATH=$HAIL_DEVBIN_DIRECTORY:$PATH
```

Refresh your shell configuration. For example, if you use `bash`:

```sh
source ~/.bashrc
```

## Building LLVM

Clone the LLVM repository and switch the version to that of the latest
stable release:

```sh
git clone https://github.com/llvm/llvm-project.git $LLVM_DIRECTORY
```

Look at the `cmake` invocation in `$HAIL_DEVBIN_DIRECTORY/hail-build-llvm` and
update it as desired.

Then, run the build for LLVM (this will take quite a while):

```sh
hail-build-llvm
```

## Building Hail's native compiler

Look at the `cmake` invocation in
`$HAIL_DEVBIN_DIRECTORY/hail-build-native-compiler` and update it as desired.

Then, run the build for Hail's native compiler:

```sh
hail-build-native-compiler
```

## Setting up your editor

You can use the Language Servers that ship with MLIR to provide you with
compiler errors, go-to-definition, and other functionality directly in your
editor. Here are some editor configurations used by members of the Hail team:

### Visual Studio Code

### Vim

### Emacs
