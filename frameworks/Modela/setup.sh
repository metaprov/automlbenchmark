#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"0.521"}
REPO=${2:-"https://github.com/metaprov/modela-python-sdk.git"}
PKG=${3:-"modela"}
API_PKG=${4:-"modelaapi"}
API_VERSION=${4:-"0.4.490"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# fix pip slowdown on wsl2
export DISPLAY=


# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true

if [[ -x "$(command -v apt-get)" ]]; then
    SUDO apt-get install -y build-essential swig
fi

PIP install --no-cache-dir -U openml
PIP install --no-cache-dir -U boto3
PIP install --no-cache-dir -U ${API_PKG}==${API_VERSION}

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}
fi

if ! command -v kubectl &> /dev/null; then
  curl -LO https://storage.googleapis.com/kubernetes-release/release/v1.23.4/bin/linux/amd64/kubectl
  chmod +x ./kubectl
  sudo mv ./kubectl /usr/local/bin/kubectl
fi

PY -c "from modela import __version__; print(__version__)" >> "${HERE}/.setup/installed"
