#!/bin/bash
python evaluate_ff3d.py \
    -s /orion/u/duyi/recon3d/gaussian/backup/data/obj_000000 \
    -m output/test_no_depth \
    --tmp_dir /orion/u/duyi/recon3d/gaussian/backup/tmp_dir \
    -o comparison.png --views 5,10,20,30