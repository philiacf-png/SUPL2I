# SUPL2I
```
bitsp/      bi-objective TSP
tritsp/     tri-objective TSP
MOCVRP/     bi-objective CVRP
testdata/   test instances
```

Requirements: Python 3.10+, PyTorch 2.x (CUDA), NumPy; `hvwfg` is used to calculate hypervolume.
Every problem directory is self-contained and follows the same command line.

## Training

```bash
cd bitsp                                                                                 # or tritsp / MOCVRP
python train.py                                                                          # 200 epochs x 10 iterations, sizes 20..100
python train.py --resume checkpoints/supl2i-bitsp-last.pt                                # continue an interrupted run
```

Progress is printed per iteration (`Epoch e/200 | iteration b/10`); every 2 epochs (20 iterations) 
the model is evaluated on 25 random validation instances of sizes 20/50/100 and the best checkpoint 
is saved as `checkpoints/supl2i-{bitsp,tritsp,bicvrp}-epoch-<E>.pt` (selection by the mean HV).
The final checkpoints are `bitsp/checkpoints/supl2i-bitsp-epoch-190.pt`,
`tritsp/checkpoints/supl2i-tritsp-epoch-190.pt` and `MOCVRP/checkpoints/supl2i-bicvrp-epoch-198.pt`.

## Inference

```bash
cd bitsp
# default: elite sharing every 200 steps, T = 1000 steps, 200 instances, 101 preference (weight) vectors
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --T 5000
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --T 1000 --aug 1      # augmentations enabled (*8)
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --elite_sharing 0     # no elite sharing
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --n_sols 51           # 51 preference (weight) vectors
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 --report front        # report the front
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 150                       # size generalisation
python eval.py --ckpt checkpoints/supl2i-bitsp-epoch-190.pt --n 100 \
    --test_data ../testdata/test_bitsp_KroAB_n100.pt --ne 1                               # a benchmark instance

cd tritsp                                                                                 # 105 preference (weight) vectors (Das-Dennis)
python eval.py --ckpt checkpoints/supl2i-tritsp-epoch-190.pt --n 100

cd MOCVRP
python eval.py --ckpt checkpoints/supl2i-bicvrp-epoch-198.pt --n 100
```

Each run prints `HV | |NDS|` averaged over the instances. HV is normalised by the reference point of the size
table, and `|NDS|` is the number of non-dominated points of the reported set. `--report` selects that set:

- `archive` (default): the non-dominated set of every solution visited by all trajectories of an instance;
- `front`: one solution per preference (weight) vector, the best one under its scalarisation that its
  trajectories hold at the end;
- `ws-archive`: one solution per preference (weight) vector, the archive point with the smallest weighted sum.

Options: 
`--n` size, 
`--ne` number of instances, 
`--T` improvement steps, 
`--n_sols` number of preference (weight) vectors, 
`--sync_every` elite-sharing period, 
`--aug 1` eight coordinate transformations, 
`--chunk` trajectories per GPU batch (reduce on small GPUs), 
`--save out.npz` per-instance results, 
`--hv_steps "0,10-100:10,200-5000:100"` hypervolume trajectory of the reported set.
