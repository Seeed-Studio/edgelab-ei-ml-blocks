#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()


while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. /tmp/data
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. out
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into coco dataset format)
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

# Disable W&B prompts
export WANDB_MODE=disabled

NUM_CLASS=`jq '.categories | length' /tmp/data/train/_annotations.coco.json`
IMG_HIGHT=`jq '.images[0].height' /tmp/data/train/_annotations.coco.json`
IMG_WIDTH=`jq '.images[0].width' /tmp/data/train/_annotations.coco.json`

cd /app/EdgeLab

echo "Training model"

python3 tools/train.py det configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    --cfg-options \
    data_root='/tmp/data' \
    epochs=$EPOCHS \
    lr=$LEARNING_RATE \
    height=$IMG_HIGHT \
    width=$IMG_WIDTH \
    num_classes=$NUM_CLASS 


echo "Training complete"

mkdir -p $OUT_DIRECTORY

# copy the model to the output directory
cp  "$(cat /app/EdgeLab/work_dirs/configs/yolov5/yolov5_tiny_1xb16_300e_coco.py/last_checkpoint)" $OUT_DIRECTORY/model.pth

python3 ./tools/torch2tflite.py /app/EdgeLab/configs/yolov5/yolov5_tiny_1xb16_300e_coco.py --checkpoint $OUT_DIRECTORY/model.pth --type float32  \
    --cfg-options \
    data_root='/tmp/data' \
    epochs=$EPOCHS \
    lr=$LEARNING_RATE \
    height=$IMG_HIGHT \
    width=$IMG_WIDTH \
    num_classes=$NUM_CLASS 

mv ./work_dirs/yolov5_tiny_1xb16_300e_coco.py/model.pth_float32.tflite $OUT_DIRECTORY/model.tflite

python3 ./tools/torch2tflite.py /app/EdgeLab/configs/yolov5/yolov5_tiny_1xb16_300e_coco.py --checkpoint $OUT_DIRECTORY/model.pth  --type int8 \
    --cfg-options \
    data_root='/tmp/data' \
    epochs=$EPOCHS \
    lr=$LEARNING_RATE \
    height=$IMG_HIGHT \
    width=$IMG_WIDTH \
    num_classes=$NUM_CLASS 

mv ./work_dirs/yolov5_tiny_1xb16_300e_coco.py/model.pth_int8.tflite $OUT_DIRECTORY/model_quantized_int8_io.tflite
