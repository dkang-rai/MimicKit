# Gitignored Data Files

Assets, motion clips and model checkpoints are **not tracked by git**. Each of these directories has a `.gitignore` containing `*`:

- `data/assets/`
- `data/motions/`
- `data/models/`
- `data/logs/`

After cloning on a new machine, set these files up manually as described below.

## Required files

| Repo path | Contents | Size | Used by |
|---|---|---|---|
| `data/assets/g1_humble/` | `g1_humble.xml` + `meshes/{collision,visual}/` (153 files) | ~46 MB | `data/envs/humble/*.yaml` |
| `data/assets/humanoid/` | `humanoid.xml`, `humanoid.usd` | small | humanoid envs |
| `data/motions/g1_humble/` | 94 `.pkl` clips, originals + `*_mirrored.pkl` | ~158 MB | `data/datasets/humble/dataset_g1_humble_*.yaml` |
| `data/motions/humanoid/` | 266 `.pkl` clips (incl. `long/`, `parc/`, `sie/`) | ~28 MB | humanoid datasets / args |
| `data/models/smp_priors/` | `smp_prior_{spinkick,lafan,locomotion}.pt` | ~63 MB | `data/agents/smp_*humanoid*_agent.yaml` |

Other upstream data, such as `data/assets/g1/`, `data/motions/g1/`, the SMPL clips and the pretrained policies in `data/models/`, comes from the archive linked in the main [README](../README.md#installation). Download and extract it into `data/` only if you need it.

## Source layout

Get the files from a machine or archive that already has them. The commands below assume `$SRC` points to a directory laid out like `data/`, without the `data/` prefix:

```
$SRC/
├── assets/{g1_humble,humanoid}/
├── motions/{g1_humble,humanoid}/
└── models/smp_priors/
```

An existing MimicKit checkout works as a source too. Set `SRC=<path-to>/MimicKit/data`.

## Manual setup

Run these commands from the repo root.

1. **Copy the files.** `rsync` merges into the existing directories and keeps their tracked `.gitignore` files:

   ```bash
   SRC=<path-or-user@host:path>
   rsync -av "$SRC/assets/"  data/assets/
   rsync -av "$SRC/motions/" data/motions/
   rsync -av "$SRC/models/"  data/models/
   ```

2. **Check that the copies match the source** (local sources only). Each command should print nothing:

   ```bash
   diff -rq "$SRC/assets"  data/assets
   diff -rq "$SRC/motions" data/motions
   diff -rq "$SRC/models"  data/models
   ```

3. **Check that every file the configs reference exists:**

   ```bash
   grep -rhoE "data/(assets|motions|models)/[^ \"']+" data/envs data/agents data/datasets args tools/diffusion_model/config \
     | sort -u \
     | while read f; do [ -e "$f" ] || echo "MISSING  $f"; done
   ```

   Paths under `data/assets/g1/`, `data/motions/g1/` and the SMPL clips are upstream data. They're expected to be missing unless you downloaded the upstream archive. Nothing under `g1_humble/` should be missing.

   To check a single dataset:

   ```bash
   grep -oE "data/motions/[^ \"']+\.pkl" data/datasets/humble/dataset_g1_humble_multi_clip.yaml \
     | while read f; do [ -e "$f" ] || echo "MISSING  $f"; done
   ```

   Mesh files referenced inside `g1_humble.xml` don't show up in this check. Copy the whole `assets/g1_humble/` directory.

## Listing the files

List every ignored file under `data/` present on this machine:

```bash
git ls-files --others --ignored --exclude-standard data | sort
```

Show a summary by directory:

```bash
git status --ignored --short data
du -sh data/assets/* data/motions/* data/models/* 2>/dev/null
```

## Saving a copy

Git doesn't keep these files, so after you add assets or motions, copy them somewhere outside the repo:

```bash
DEST=<path-or-user@host:path>
rsync -av --exclude .gitignore data/assets/  "$DEST/assets/"
rsync -av --exclude .gitignore data/motions/ "$DEST/motions/"
rsync -av --exclude .gitignore data/models/  "$DEST/models/"
```

The result uses the same layout as `$SRC` above, so it can serve as the source on the next machine.
