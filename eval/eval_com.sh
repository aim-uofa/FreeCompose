export BACK_ORI_DIR="path-to-background-original-images"
export FORE_ORI_DIR="path-to-foreground-original-images"
export MASK_DIR="path-to-mask-images"
export GEN_DIR="path-to-generated-images"

python eval/base_score.py \
    --ori_dir $BACK_ORI_DIR \
    --gen_dir $GEN_DIR \
    --mask_dir $MASK_DIR

python eval/clip_score.py \
    --ori_dir $BACK_ORI_DIR \
    --gen_dir $GEN_DIR

python eval/clip_score.py \
    --ori_dir $FORE_ORI_DIR \
    --gen_dir $GEN_DIR \
    --foreground true

python eval/dino_score.py \
    --ori_dir $BACK_ORI_DIR \
    --gen_dir $GEN_DIR

python eval/dino_score.py \
    --ori_dir $FORE_ORI_DIR \
    --gen_dir $GEN_DIR \
    --foreground true

python eval/fid_score.py \
    $BACK_ORI_DIR $GEN_DIR

python eval/qs_score.py $GEN_DIR \
    --gmm_path data/coco2017_gmm_k20