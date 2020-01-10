#!/usr/bin/env bash

#echo "fps=1"
#python eval.py  --file_list list.txt --data_dir PartoImg/test_fps/pcb_fps_1

echo "fps=2"
python eval.py  --file_list list.txt --data_dir PartoImg/test_fps/pcb_fps_2 --upscale_factor 2 --model weights/RBPN_2x.pth

echo "fps=4"
python eval.py  --file_list list.txt --data_dir PartoImg/test_fps/pcb_fps_4  --upscale_factor 2 --model weights/RBPN_2x.pth

echo "fps=10"
python eval.py  --file_list list.txt --data_dir PartoImg/test_fps/pcb_fps_10 --upscale_factor 2 --model weights/RBPN_2x.pth

echo "fps=30"
python eval.py  --file_list list.txt --data_dir PartoImg/test_fps/pcb_fps_30 --upscale_factor 2 --model weights/RBPN_2x.pth

echo "fps=60"
python eval.py  --file_list list.txt --data_dir PartoImg/test_fps/pcb_fps_60 --upscale_factor 2 --model weights/RBPN_2x.pth
