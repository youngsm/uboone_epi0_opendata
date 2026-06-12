#!/bin/sh
# Build the actual larreco Solver.cxx as a Python module.
# Sources Solver.{h,cxx} QuadExpr.{h,cxx} are vendored UNMODIFIED from
# github.com/LArSoft/larreco @ 6c4c0fd918a5577feeaa3865e5229de7b36f075c
# (verify: sha256sum -c SOURCES.sha256). Requires: g++, pybind11.
cd "$(dirname "$0")"
# build against the interpreter you actually run (override: PYTHON=... build.sh)
PYTHON="${PYTHON:-python}"

# checksum tool differs across platforms (linux: sha256sum, macOS: shasum)
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c SOURCES.sha256 || exit 1
else
    shasum -a 256 -c SOURCES.sha256 || exit 1
fi

# macOS links python extensions with two-level namespace resolved at load
# time; linux uses a plain shared object. -O3 + -ffast-math kept off to
# preserve the <1e-12 equivalence vs the Python port.
case "$(uname -s)" in
    Darwin) LDFLAGS="-dynamiclib -undefined dynamic_lookup" ;;
    *)      LDFLAGS="-shared -fPIC" ;;
esac

EXT=$("$PYTHON" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
g++ -O3 -std=c++17 $LDFLAGS solver_bind.cpp Solver.cxx QuadExpr.cxx \
    -I. $("$PYTHON" -m pybind11 --includes) \
    -o "spsolver_cpp$EXT"
echo "built: spsolver_cpp$EXT"
