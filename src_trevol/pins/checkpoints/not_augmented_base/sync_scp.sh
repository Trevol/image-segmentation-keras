#!/usr/bin/env bash

SRV="trevol@192.168.0.115"
CURRENT_DIR="$(pwd)"
UPPER_DIR="$CURRENT_DIR/.."

scp -r $SRV:$CURRENT_DIR $UPPER_DIR
