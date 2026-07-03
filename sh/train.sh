WANDB_NAME=default_DIT_bs30x3_node8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
ABC_CACHE=/dev/shm/abc_cache \
uv run torchrun --standalone --nproc-per-node 8 train.py \
  --log-wandb \
  --wandb-project abc_minimal \
  --mixture-preset bottles \
  --grad-accum-steps 3 \
  --output-dir outputs/checkpoints/default_DIT_grad_accum \
  2>&1 | tee train.log