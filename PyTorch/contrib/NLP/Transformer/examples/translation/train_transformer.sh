mkdir -p checkpoints/transformer_iwslt_de_en

SDAA_VISIBLE_DEVICES=0,1 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir checkpoints/transformer_iwslt_de_en \
    --log-format json \
    --log-interval 200 \
    --log-file checkpoints/transformer_iwslt_de_en/log.log \
    --max-epoch 20 \
    --amp \

fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/transformer_iwslt_de_en/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe