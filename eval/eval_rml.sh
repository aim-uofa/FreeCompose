export ORI_DIR="path-to-original-images"
export MASK_DIR="path-to-mask-images"
export GEN_DIR="path-to-generated-images"

python eval/base_score.py \
    --ori_dir $ORI_DIR \
    --gen_dir $GEN_DIR \
    --mask_dir $MASK_DIR \
    --task rml

python eval/fid_score.py \
    $ORI_DIR $GEN_DIR

python eval/lpips_score.py \
    --ori_dir $ORI_DIR \
    --gen_dir $GEN_DIR