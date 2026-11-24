# Mac preparation and Terrawulf inversion

Use the combined `codex/mac-terra-workflow` branch on both machines. It contains
empirical receiver/source STF selection and reproducible shared receiver window
sweeps, the validated forward-model speedup, and shared/per-trace manual STF
support. Select `mode: per_trace` to export one STF per retained seismogram.

## Mac

The normal checkout is `~/Documents/Research/TAPIR`. In the `tapir` environment,
install it with `python -m pip install -e . --no-deps`. This points the command
line tools and Python imports at this checkout.

Keep machine paths in ignored `configs/workspace.local.yaml`. Personal prep
configs can use ignored names such as `configs/prep_receiver.local.yaml` and
`configs/prep_source.local.yaml`, with `workspace: workspace.yaml`. The tracked
example configs remain portable. Download and preprocess locally, inspect QC
and STF selection, and retain selection metadata for replay.

Copy the entire prepared dataset directory to Terra, including waveform,
noise/covariance, geometry, time, STF, manifest, selection, and trace-ID files.
Copy a campaign YAML separately and adjust its `data_root` and `runs_root` for
Terra. A Mac absolute path in a YAML will not work on Terra. For example:

```bash
scp -r /path/to/prepared/EVENT ziqi@terrawulf.anu.edu.au:~/projects/TAPIR/RealData/
scp /path/to/campaign.yaml ziqi@terrawulf.anu.edu.au:~/projects/TAPIR/configs/
```

## Terra

Do not switch or update this checkout while jobs using it are queued or running.
Historical frozen sweep bundles keep their own code and are unaffected.

For the initial transition from `codex/trace-specific-stf`, first inspect
`git status`. Preserve any tracked edits before switching; untracked personal
configs and data can remain. Then:

```bash
cd ~/projects/TAPIR
git fetch origin
git switch --track origin/codex/mac-terra-workflow
conda activate tapir
python -m pip install -e . --no-deps
git rev-parse --short HEAD
python -c 'import vespainv; print(vespainv.__file__)'
```

Later code updates need only `git pull --ff-only` on this branch. Reinstall if
packaging or dependency requirements change. Data and configs still travel
separately. Validate a prepared 3C receiver dataset before submission:

```bash
tapir validate-data /home/ziqi/projects/TAPIR/RealData/EVENT --cdopt 3 --manual-stf
```

Choose `--cdopt` to match the campaign, and omit `--manual-stf` for generated
wavelets. Start with a short campaign before a long run. Submit with absolute
paths (replace `campaign.yaml`):

```bash
mkdir -p ~/projects/TAPIR/logs
qsub -o /home/ziqi/projects/TAPIR/logs/ \
     -e /home/ziqi/projects/TAPIR/logs/ \
     -v TAPIR_CONFIG=/home/ziqi/projects/TAPIR/configs/campaign.yaml \
     scripts/hpc/terrawulf.pbs
```

The PBS launcher logs the Git revision and uses the allocated CPUs with one
numerical-library thread per chain. Use distinct output run names to preserve
previous results. Keep the launch log, config, prepared data/metadata, and
`stf_used.csv` with results so later comparisons identify the exact inputs.
