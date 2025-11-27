#!/bin/bash

# - - - - - configure paths as needed - - - - -
DATA_DIR="data"
# - - - - - - - - - - - - - - - - - - - - - - - 

# build absolute paths
SCRIPT_DIR=`dirname "$0"`
ABSOLUTE_SCRIPT_DIR=`( cd "$SCRIPT_DIR" && pwd )`
ABSOLUTE_DATA_DIR="$ABSOLUTE_SCRIPT_DIR/$DATA_DIR"

# export environment variables
export EUROCROPS_DATA_DIR=$ABSOLUTE_DATA_DIR

# run initialization
eurocropsml-cli datasets eurocrops download