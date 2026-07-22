# Historical Runpod A40 + Network Volume: BioASQ Semantic Entropy

> Status (2026-07-13): this temporary Runpod recovery path is complete. The
> archived required batch is under
> `server_results/bioasq_se_runpod_required_results_20260713.tar.gz`; future
> experiments run on Isambard. Retain this document only to reproduce or audit
> the completed A40 batch, not as the active launch guide.

This guide is for a **new Runpod A40 pod** with a portable Network Volume.
The accompanying upload archive contains the maintained code, public BioASQ
Task B data, tests, and this guide. It deliberately excludes old outputs,
Hugging Face caches, virtual environments, Git history, and restricted
ArchEHR-QA data.

The current upload archive retains its existing `archehr_sebaseline` name as a
path/provenance contract. Future new BioASQ artifacts follow
`docs/naming_policy.md`; do not rename this archive or the package directory.

## Pod and storage configuration

Choose a Secure Cloud PyTorch/CUDA pod with **1x A40 (48 GB VRAM)**. The
precise host RAM and vCPU count vary by data centre; the previous 48 GB A40
run completed this workload, so no L40S-only resource is required.

| Setting | Choose | Why |
| --- | ---: | --- |
| Container Disk | **40 GB** | Image, temporary files, and pip cache only. It is local and erased when the pod stops or restarts. |
| Network Volume | **100 GB** | Persistent /workspace: project, Gemma/NLI/Qwen caches, public data, logs, and outputs. It is portable to a new pod. |

First select a data centre where the console offers an A40 in Secure Cloud,
then create the Network Volume in that **same** data centre and attach it
during pod creation. The volume mounts at /workspace and replaces the ordinary
volume disk. It can grow later but cannot shrink. It persists after the pod is
terminated, but one attached volume constrains the pod to its data centre. See
the official [storage comparison](https://docs.runpod.io/pods/storage/types)
and [Network Volume guide](https://docs.runpod.io/storage/network-volumes).

All persistent project state below lives under /workspace. Do **not** put model
caches or experiment outputs in /tmp or a home-directory cache.

## Scope: no duplicate experiments

The completed primary A40 result is preserved locally as
server_results/runpod_bioasq_summary_gemma3_12b_100x10/, including raw
artifacts and its lightweight BioASQ evaluation. It is **not** included in the
upload archive and must **not** be rerun.

The old Qwen 5x3 NLI preflight and Gemma training13b summary 100x10 primary
run are already validated. On this new A40, run setup once and then launch the
single remaining-experiments batch. The first Golden job is the runtime
validation; if it fails, the batch stops without spending time on later jobs.

## Upload and first-time setup

Upload archehr_sebaseline_runpod_a40_network_20260713.tar.gz to /workspace,
then run:

~~~bash
cd /workspace
tar -xzf archehr_sebaseline_runpod_a40_network_20260713.tar.gz
cd archehr_sebaseline_runpod_a40_network_20260713/archehr_sebaseline
bash scripts/runpod_setup.sh
source .venv_runpod/bin/activate
hf auth login
hf auth whoami
~~~

Setup creates .venv_runpod with --system-site-packages, retains the
CUDA-enabled PyTorch from the Runpod image, installs the remaining Python
requirements, checks CUDA, and runs the 47-test local suite. It should report
the A40 name and roughly 48 GB of GPU memory.

The batch script activates `.venv_runpod` internally before every model job,
so activating it again inside tmux is optional. Keep it activated for manual
commands such as `hf`, direct Python calls, or ad-hoc inspection.

The project keeps HF_HOME at
/workspace/archehr_sebaseline_runpod_a40_network_20260713/archehr_sebaseline/.cache/huggingface,
so model downloads survive pod termination as long as the Network Volume is
attached to the next pod.

## One unattended batch for all required experiments

Start tmux and launch the batch once. It runs the four required jobs
sequentially:

1. Golden summary, Gemma, requested 100x10 (the official Golden 13B summary
   subset currently supplies 80 compatible questions, so this run is 80x10)
2. Golden factoid, Gemma, 50x10
3. Golden list, Gemma, 50x10
4. Training13b summary sampling repeat, Gemma, 50x10, seed 47

Each job already performs its Level 4 health check. Immediately after a
successful health check, the batch runs the lightweight BioASQ evaluator. On
any command failure, set -euo pipefail stops the batch; it does **not**
continue silently to later experiments. All terminal output is saved below
outputs/batch_logs/.

`MAX_EXAMPLES` is a requested upper bound, not an assertion that the source
contains that many compatible questions. The runner now counts
`examples.jsonl` after generation and validates the matching generation count.
Therefore the existing Golden-summary output directory retains its historical
`..._100x10` name, but its correct health-check expectation is 80 examples and
800 generations; it does not need to be rerun.

~~~bash
tmux new -s bioasq_batch
bash scripts/runpod_remaining_experiments.sh
~~~

Detach with Ctrl-b, then d. Reconnect only when convenient:

~~~bash
tmux attach -t bioasq_batch
~~~

No periodic manual checking is required. To check progress without interrupting
anything, reconnect to tmux or inspect the current log:

~~~bash
tail -f outputs/batch_logs/*.log
~~~

If the pod or terminal is interrupted, start the same batch command again.
By default it skips a run only when both its health check reports PASS and its
BioASQ evaluation summary exist; incomplete jobs are rerun with --overwrite.

## Optional Qwen comparison

The Qwen Golden-summary 50x10 comparison is intentionally excluded from the
default batch, because it is supplementary rather than required evidence. If
all four required jobs finish and budget remains, start it explicitly:

~~~bash
RUN_OPTIONAL_QWEN=1 bash scripts/runpod_remaining_experiments.sh
~~~

The already completed four jobs will be skipped, so this invocation runs only
the optional Qwen comparison.

## Batch outputs and interpretation

For every completed run the batch writes:

- summary.txt and health_check.txt: structural and CUDA/NLI verification
- bioasq_eval/: lightweight quality scores, SE AUROC table, and rejection curve
- outputs/batch_logs/: durable terminal logs

The evaluator is a transparent local approximation of BioASQ metric families,
**not** the official BioASQ evaluation service. Do not modify the generation
prompt merely to raise its lexical score: a mixture of stronger and weaker
answers is needed to evaluate Semantic Entropy filtering.

## End of pod: preserve and move data

Before terminating, verify that code, cache, logs, and outputs are all under
/workspace. The Network Volume survives termination and can be attached to a
future compatible A40 pod in the same data centre. Still download completed
output archives to local storage as an independent backup.

Archive the complete required batch, not just one directory. The helper refuses
to create a misleading partial archive: all four required runs must have a
passing health check and BioASQ evaluation summary. It also includes
`batch_logs/` and includes the optional Qwen run when present.

~~~bash
cd /workspace/archehr_sebaseline_runpod_a40_network_20260713/archehr_sebaseline
bash scripts/package_runpod_bioasq_results.sh
~~~

The command prints the verified archive path under `outputs/`; download that
single archive before terminating the pod. For an already-running pod that
does not yet have this helper, use the equivalent command below (replace only
the date in the output filename if desired):

~~~bash
cd /workspace/archehr_sebaseline_runpod_a40_network_20260713/archehr_sebaseline
tar -czf outputs/bioasq_se_runpod_required_results_20260713.tar.gz \
  outputs/runpod_bioasq_golden_summary_gemma3_12b_100x10 \
  outputs/runpod_bioasq_golden_factoid_gemma3_12b_50x10 \
  outputs/runpod_bioasq_golden_list_gemma3_12b_50x10 \
  outputs/runpod_bioasq_train_summary_gemma3_12b_50x10_seed47 \
  outputs/batch_logs
tar -tzf outputs/bioasq_se_runpod_required_results_20260713.tar.gz > /dev/null && echo "archive verified"
~~~
