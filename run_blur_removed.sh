#!/usr/bin/env bash

echo "fps=4"
python eval.py --file_list fps_4.txt --data_dir ./PartoImg/blur_removed/fps_4

echo "fps=10"
python eval.py --file_list fps_10.txt --data_dir ./PartoImg/blur_removed/fps_10

echo "fps=60"
python eval.py --file_list fps_60.txt --data_dir ./PartoImg/blur_removed/fps_60