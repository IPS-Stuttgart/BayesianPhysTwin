# PokeFlex missing-five V6 execution

## Registered candidate

The V6 execution lock augments, but never replaces, a sealed V5 prediction.
For every target take, the V5 stage and prediction must first be produced with
`scripts/held/run_pokeflex_missing5_v5.py`. V6 then replays only the authorized
prefix inputs to compute the five causal scale features and applies the frozen
source model.

- Execution protocol SHA-256: `e875495295acc8de4b7da70cdcaff9947838b8e92f1785dbbe32f9fc0c67e78b`
- Execution protocol file SHA-256: `11cf0cabf40760048f05ffee93a3cdd104b80fe90306d8ce0a3cfe0da4c048a3`
- Parent V5 execution SHA-256: `1bc0a3486c0b937000772fc74bfcab2ed4a4dd34f2d90d0199440bc043a59f7a`
- Causal scale model SHA-256: `4b6835f6ab57787be007855141081a3a10cea30eba736d47863b68fa7acf6ffa`
- Source result SHA-256: `e7f17d5bd9045a3634e4f07b32ae9217ea8432ddd24ed387ab1c3512dd18483f`

No official target outcome was used to choose the model or lock the runner.
The exact five author-provided archives were unavailable when this protocol
was registered.

## Procedure

For each of the five registered archives:

1. Run the V5 `stage` command. It extracts only `robot_data.json`, the two
   camera parameter files, depth frames preceding each prediction, and the one
   task-authorized template mesh.
2. Run the V5 `predict` command and validate its seal.
3. Run the V6 `augment` command with that same stage and sealed V5 prediction.
   The augmentation refuses a replay whose support mask differs from V5.
4. After all five V6 seals exist at one clean revision, run the V6 `barrier`
   command.
5. Run `score` only after the complete barrier authorizes target-mesh access.

The V6-specific commands are:

```bash
python scripts/held/run_pokeflex_missing5_v6.py \
  --source-manifest /path/to/source-manifest.json \
  augment /path/to/v5-stage /path/to/v5-prediction /path/to/v6-prediction

python scripts/held/run_pokeflex_missing5_v6.py \
  --source-manifest /path/to/source-manifest.json \
  barrier /path/to/v5-predictions /path/to/v6-predictions /path/to/v6-barrier.json

python scripts/held/run_pokeflex_missing5_v6.py \
  --source-manifest /path/to/source-manifest.json \
  score /path/to/exact-archives /path/to/v5-predictions \
  /path/to/v6-predictions /path/to/v6-barrier.json /path/to/v6-result.json
```

## Gates

V6 advances only if the five-target object-balanced mean is strictly below V5,
no target object regresses, and the 97.5% paired-object bootstrap upper bound
for V6 minus V5 is at most zero. The combined official-18 frame mean must also
be strictly below V5 and below the published `6.498` mm reference, with a
nonpositive paired-object bootstrap upper bound.

Unsupported frames are byte-identical to the released checkpoint. Supported
but rejected frames are byte-identical to V5. Pizza, Pillow, and Sponge always
use V5. No replacement, target adaptation, or post-outcome tuning is allowed.
