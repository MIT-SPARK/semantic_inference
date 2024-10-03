#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

pushd $(dirname $SCRIPT_DIR)

find . -iname *.h -o -iname *.cpp | xargs python3 $SCRIPT_DIR/check_license.py add
find . -iname '*.py' | xargs python3 $SCRIPT_DIR/check_license.py add

popd
