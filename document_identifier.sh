#!/bin/bash

if [ ! "$#" -eq 1 ]; then
  echo "Usage: $0 <input>"
  exit 1
fi

INPUT=$1

source /opt/intel/openvino/bin/setupvars.sh 

DWL_CMD="$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py -o ./models/"

TXT_DETECT_MODEL=text-spotting-0002-detector
TXT_ENC_MODEL=text-spotting-0002-recognizer-encoder
TXT_DEC_MODEL=text-spotting-0002-recognizer-decoder

THRESHOLD=0.8

mkdir -p ./models/

[ ! -d ./models/intel/$TXT_DETECT_MODEL ] && $DWL_CMD --name $TXT_DETECT_MODEL 
[ ! -d ./models/intel/$TXT_ENC_MODEL ] && $DWL_CMD --name $TXT_ENC_MODEL 
[ ! -d ./models/intel/$TXT_DEC_MODEL ] && $DWL_CMD --name $TXT_DEC_MODEL 

python3 ./document_identifier.py \
	-m_m ./models/intel/$TXT_DETECT_MODEL/FP16/$TXT_DETECT_MODEL.xml \
	-m_te ./models/intel/$TXT_ENC_MODEL/FP16/$TXT_ENC_MODEL.xml \
	-m_td ./models/intel/$TXT_DEC_MODEL/FP16/$TXT_DEC_MODEL.xml \
	-pt $THRESHOLD \
	-i $INPUT 


