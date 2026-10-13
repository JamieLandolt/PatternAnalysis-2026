#!/bin/bash
# Submit an experiment group to Slurm: every model x seed x setting, one job each.
#   ./run_experiments.sh transform      # return-transform ablation (zscore/clip/quantile)
#   ./run_experiments.sh step           # grid step size (0.5/1/5/10 s)
#   ./run_experiments.sh loss           # TimeGAN loss weights, feature discriminator, cosine LR
#   ./run_experiments.sh depth          # difference moment-matching loss for depth persistence
# Results land in outputs/<group>/<setting>/<model>_s<seed>/ (gitignored).
export PATH=/opt/slurm/bin:$PATH
GROUP=${1:?usage: run_experiments.sh <transform|step>}
SEEDS=${SEEDS:-"0 1 2"}

submit() {  # submit <setting-name> <model> <seed> <extra train.py args...>
    local setting=$1 model=$2 seed=$3; shift 3
    OUT_DIR=outputs/$GROUP/$setting/${model}_s$seed SEED=$seed \
        sbatch --parsable --job-name=${GROUP:0:3}_${setting}_${model:0:2}$seed \
        slurm_train.sh $model "$@"
}

for seed in $SEEDS; do
  for model in timegan baseline; do
    # loss/depth settings only change TimeGAN training, so the baseline is skipped
    [[ $model == baseline && ( $GROUP == loss || $GROUP == depth ) ]] && continue
    case $GROUP in
      transform)
        for t in zscore clip quantile; do submit $t $model $seed --return-transform $t; done ;;
      step)
        for st in 0.5 1 5 10; do submit step$st $model $seed --step $st; done ;;
      loss)
        submit default  $model $seed
        submit mom10    $model $seed --w-moment 10
        submit mom0     $model $seed --w-moment 0
        submit sup10    $model $seed --w-sup 10
        submit featadv  $model $seed --feature-adv
        submit cosine   $model $seed --lr-schedule cosine --epochs 150 ;;
      depth)
        submit diff10   $model $seed --w-diff 10
        submit diff100  $model $seed --w-diff 100 ;;
      *) echo "unknown group $GROUP"; exit 1 ;;
    esac
  done
done
