#!/bin/bash

if [ ! "$#" -eq 1 ]; then
  echo "Usage: $0 <input>"
  exit 1
fi

INPUT=$1

pip3 freeze | grep -q Flask || pip3 install Flask --user
pip3 freeze | grep -q json2html || pip3 install json2html --user

source /opt/intel/openvino/bin/setupvars.sh 

DWL_CMD="$INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py -o ./models/"

TXT_DETECT_MODEL=text-spotting-0002-detector
TXT_ENC_MODEL=text-spotting-0002-recognizer-encoder
TXT_DEC_MODEL=text-spotting-0002-recognizer-decoder
FACE_DETECT_MODEL=face-detection-adas-0001
FACIAL_LAND_MARKS=facial-landmarks-35-adas-0002

THRESHOLD=0.7

mkdir -p ./models/

[ ! -d ./models/intel/$TXT_DETECT_MODEL ] && $DWL_CMD --name $TXT_DETECT_MODEL 
[ ! -d ./models/intel/$TXT_ENC_MODEL ] && $DWL_CMD --name $TXT_ENC_MODEL 
[ ! -d ./models/intel/$TXT_DEC_MODEL ] && $DWL_CMD --name $TXT_DEC_MODEL 
[ ! -d ./models/intel/$FACE_DETECT_MODEL ] && $DWL_CMD --name $FACE_DETECT_MODEL
[ ! -d ./models/intel/$FACIAL_LAND_MARKS ] && $DWL_CMD --name $FACIAL_LAND_MARKS


python3 ./document_identifier.py \
	-m_m ./models/intel/$TXT_DETECT_MODEL/FP16/$TXT_DETECT_MODEL.xml \
	-m_te ./models/intel/$TXT_ENC_MODEL/FP16/$TXT_ENC_MODEL.xml \
	-m_td ./models/intel/$TXT_DEC_MODEL/FP16/$TXT_DEC_MODEL.xml \
	-m_fd ./models/intel/$FACE_DETECT_MODEL/FP16/$FACE_DETECT_MODEL.xml \
	-m_lm ./models/intel/$FACIAL_LAND_MARKS/FP16/$FACIAL_LAND_MARKS.xml \
	-c ./config.js \
	-pt $THRESHOLD \
	-i $INPUT 


