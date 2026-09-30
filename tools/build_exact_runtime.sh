#!/usr/bin/env bash
# Reproducible source identities; dependency packages come from the runner OS.
set -euo pipefail
root=$(realpath -m "${1:?build root required}")
mkdir -p "$root"
fetch() {
  local name=$1 sha=$2
  if [ ! -d "$root/$name/.git" ]; then
    git init "$root/$name"
    git -C "$root/$name" remote add origin "https://github.com/scipopt/$name.git"
  fi
  git -C "$root/$name" fetch --depth 1 origin "$sha"
  git -C "$root/$name" checkout --detach "$sha"
  test "$(git -C "$root/$name" rev-parse HEAD)" = "$sha"
}
fetch soplex 13e2ab2467e0016d02116802ac4dc7a89560dbc1
fetch scip d409edf9f6f25aeab3b849c125826848299953c3
fetch vipr 30f2951d1e90e47afa821bdd1b12b82246656c42
cmake -S "$root/soplex" -B "$root/soplex-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/install" -DGMP=ON \
  -DBOOST=ON -DMPFR=ON
cmake --build "$root/soplex-build" --parallel 2
cmake --install "$root/soplex-build"
cmake -S "$root/scip" -B "$root/scip-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/install" \
  -DCMAKE_PREFIX_PATH="$root/install" -DCMAKE_INSTALL_RPATH="$root/install/lib" \
  -DEXACTSOLVE=ON -DGMP=ON -DMPFR=ON -DBOOST=ON \
  -DPAPILO=OFF -DZIMPL=OFF -DIPOPT=OFF -DAMPL=OFF -DREADLINE=OFF
cmake --build "$root/scip-build" --parallel 2
cmake --install "$root/scip-build"
cmake -S "$root/vipr/code" -B "$root/vipr-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$root/install" -DVIPRCOMP=OFF
cmake --build "$root/vipr-build" --target viprchk --parallel 2
mkdir -p "$root/install/bin"
cp "$root/vipr-build/viprchk" "$root/install/bin/viprchk"
