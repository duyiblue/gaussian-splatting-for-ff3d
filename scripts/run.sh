#!/bin/bash
python ../train.py \
    -s /orion/u/duyi/recon3d/gaussian/backup/data/obj_000000 \
    -m output/test_no_depth \
    --tmp_dir /orion/u/duyi/recon3d/gaussian/backup/tmp_dir \
    --test_iterations 100 5000 10000 20000 30000 \
    --save_iterations 100 5000 10000 20000 30000 \
    --disable_viewer \
    --checkpoint_iterations 100 5000 10000 20000 30000