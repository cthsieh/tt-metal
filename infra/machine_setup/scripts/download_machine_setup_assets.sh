#!/usr/bin/env bash

set -eo pipefail

TT_METAL_HOME=$(git rev-parse --show-toplevel)
ASSETS_DIR=$TT_METAL_HOME/infra/machine_setup/assets

SERVER=10.250.37.82
GS_TT_SMI_SERVER_LOCATION=/home/software/syseng/bin/tt-smi/gs/tt-smi_2022-12-05-74801e089fb2e564
WH_TT_SMI_SERVER_LOCATION=/home/software/syseng/bin/tt-smi/wh/tt-smi-wh_2023-02-07-eda6cb21c5788763
GS_TT_FLASH_SERVER_LOCATION=/home/software/syseng/bin/tt-flash/gs/tt-flash_2022-09-06-fae5785cae3807a6
WH_TT_FLASH_SERVER_LOCATION=/home/software/syseng/bin/tt-flash/wh/tt-flash_7.8.2.0_2023-03-29-ea858ffb9a5c19e3
TT_DRIVER_SERVER_LOCATION=/home/software/syseng/bin/tt-kmd/install_ttkmd_1.19.bash
GS_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/gs
WH_TT_SMI_LOCAL_FOLDER=$ASSETS_DIR/tt_smi/wh
GS_TT_SMI_LOCAL_LOCATION=$GS_TT_SMI_LOCAL_FOLDER/tt-smi
WH_TT_SMI_LOCAL_LOCATION=$WH_TT_SMI_LOCAL_FOLDER/tt-smi
GS_TT_FLASH_LOCAL_FOLDER=$ASSETS_DIR/tt_flash/gs
WH_TT_FLASH_LOCAL_FOLDER=$ASSETS_DIR/tt_flash/wh
GS_TT_FLASH_LOCAL_LOCATION=$GS_TT_FLASH_LOCAL_FOLDER/tt-flash
WH_TT_FLASH_LOCAL_LOCATION=$WH_TT_FLASH_LOCAL_FOLDER/tt-flash
TT_DRIVER_LOCAL_FOLDER=$ASSETS_DIR/tt_driver
TT_DRIVER_LOCAL_LOCATION=$TT_DRIVER_LOCAL_FOLDER/install_ttkmd.bash

rm -rf $ASSETS_DIR
mkdir -p $GS_TT_SMI_LOCAL_FOLDER
mkdir -p $WH_TT_SMI_LOCAL_FOLDER
mkdir -p $GS_TT_FLASH_LOCAL_FOLDER
mkdir -p $WH_TT_FLASH_LOCAL_FOLDER
mkdir -p $TT_DRIVER_LOCAL_FOLDER

scp $SERVER:"$GS_TT_SMI_SERVER_LOCATION" $GS_TT_SMI_LOCAL_LOCATION
scp $SERVER:"$WH_TT_SMI_SERVER_LOCATION" $WH_TT_SMI_LOCAL_LOCATION
scp $SERVER:"$GS_TT_FLASH_SERVER_LOCATION" $GS_TT_FLASH_LOCAL_LOCATION
scp $SERVER:"$WH_TT_FLASH_SERVER_LOCATION" $WH_TT_FLASH_LOCAL_LOCATION
scp $SERVER:"$TT_DRIVER_SERVER_LOCATION" $TT_DRIVER_LOCAL_LOCATION
