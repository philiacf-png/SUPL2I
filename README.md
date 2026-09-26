# SUPL2I
```
bitsp/      bi-objective TSP
tritsp/     tri-objective TSP
MOCVRP/     bi-objective CVRP
testdata/   test instances
```

Requirements: Python 3.10+, PyTorch 2.x (CUDA), NumPy; `hvwfg` is optional (exact hypervolume;
a pure-numpy fallback is built in), `tensorboard` is optional (training curves).
Every problem directory is self-contained (`env.py`, `layers.py`, `model.py`, `train.py`, `eval.py`,
`make_seeds.py`) and follows the same command line.

## Training

```bash
cd bitsp                                          # or tritsp / MOCVRP
python train.py                                   # 200 epochs x 10 iterations, sizes 20..100
python train.py --resume checkpoints/supl2i-bitsp-last.pt   # continue an interrupted run
```

Progress is printed per iteration (`Epoch e/200 | iteration b/10`); every 100 iterations the
model is evaluated on 25 random validation instances of sizes 20/50/100 and the best checkpoint 
is saved as `checkpoints/supl2i-{bitsp,tritsp,bicvrp}-epoch-{E}.pt` (selection by the mean HV).
The final checkpoints are `bitsp/checkpoints/supl2i-bitsp-epoch-190.pt`,
`tritsp/checkpoints/supl2i-tritsp-epoch-{E}.pt` and `MOCVRP/checkpoints/supl2i-bicvrp-epoch-{E}.pt`.

## Inference

```bash
cd bitsp
# default: elite sharing every 200 steps, T = 1000 steps, 200 instances
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --T 5000
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --T 1000 --aug 1      # augmentations enabled (*8)
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --elite_sharing 0     # no elite sharing
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 150                       # size generalisation
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 \
    --test_data ../testdata/test_bitsp_KroAB_n100.pt --ne 1                                # a benchmark instance

cd tritsp                                         # 105 preference vectors (Das-Dennis)
python eval.py --ckpt checkpoints/supl2i-tritsp-epoch-{E}.pt --n 100

cd MOCVRP                                         # feasible solutions only; every reported ring is re-verified
python eval.py --ckpt checkpoints/supl2i-bicvrp-epoch-{E}.pt --n 100
```

Options: `--n` size, `--ne` number of instances, `--T` improvement steps, `--sync_every` elite-sharing
period, `--aug 1` eight coordinate transformations, `--chunk` trajectories per GPU batch (reduce on small
GPUs), `--save out.npz` per-instance results, `--hv_steps "0,10-100:10,200-5000:100"` hypervolume trajectory.
