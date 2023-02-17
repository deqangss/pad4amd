#!/bin/bash

for i in {10,20,30}
do
    python -m examples.mimicry_test --trials $i --model "md_dnn" --model_name "20220629-224059"
    python -m examples.mimicry_test --trials $i --model "md_at_pgd" --model_name "20220711-103657"
    python -m examples.mimicry_test --trials $i --model "md_at_ma" --model_name "20220711-110103"
    python -m examples.mimicry_test --trials $i --model "amd_kde" --model_name "20220629-224059" --oblivion
    python -m examples.mimicry_test --trials $i --model "amd_dla" --model_name "20220711-122226" --oblivion
    python -m examples.mimicry_test --trials $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
    python -m examples.mimicry_test --trials $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
    python -m examples.mimicry_test --trials $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done
#
for i in {1,2,5}
do
   python -m examples.grosse_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.grosse_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.grosse_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.grosse_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done
for i in {10..120..10}
do
   python -m examples.grosse_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.grosse_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.grosse_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.grosse_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.grosse_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done


for i in {1,2,5}
do
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "md_dnn" --model_name "20220629-224059"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done
for i in {10..120..10}
do
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "md_dnn" --model_name "20220629-224059"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done

for i in {1,2,5}
do
   python -m examples.bga_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.bga_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.bga_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.bga_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.bga_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.bga_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.bga_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.bga_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done
for i in {10..120..10}
do
   python -m examples.bga_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.bga_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.bga_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.bga_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.bga_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.bga_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.bga_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.bga_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done

for i in {1,2,5}
do
   python -m examples.bca_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.bca_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.bca_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.bca_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.bca_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.bca_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.bca_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.bca_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done
for i in {10..120..10}
do
   python -m examples.bca_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.bca_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.bca_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.bca_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.bca_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.bca_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.bca_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.bca_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done

for i in {1,2,5}
do
   python -m examples.pgdl1_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.pgdl1_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.pgdl1_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.pgdl1_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done
for i in {10..120..10}
do
   python -m examples.pgdl1_test --steps $i --model "md_dnn" --model_name "20220629-224059"
   python -m examples.pgdl1_test --steps $i --model "md_at_pgd" --model_name "20220711-103657"
   python -m examples.pgdl1_test --steps $i --model "md_at_ma" --model_name "20220711-110103"
   python -m examples.pgdl1_test --steps $i --model "amd_kde" --model_name "20220629-224059" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_dla" --model_name "20220711-122226" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
   python -m examples.pgdl1_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done


python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "md_dnn" --model_name "20220629-224059"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "md_at_pgd" --model_name "20220711-103657"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "md_at_ma" --model_name "20220711-110103"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_kde" --model_name "20220629-224059" --oblivion
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_dla" --model_name "20220711-122226" --oblivion
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_icnn" --model_name "20220711-130831" --oblivion
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_pad_ma" --model_name "20230124-082522" --oblivion


python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "md_dnn" --model_name "20220629-224059"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "md_at_pgd" --model_name "20220711-103657"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "md_at_ma" --model_name "20220711-110103"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_kde" --model_name "20220629-224059" --oblivion
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_dla" --model_name "20220711-122226" --oblivion
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_icnn" --model_name "20220711-130831" --oblivion
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_pad_ma" --model_name "20230124-082522" --oblivion


for i in {1,5};
do
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "md_dnn" --model_name "20220629-224059"
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "md_at_pgd" --model_name "20220711-103657"
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "md_at_ma" --model_name "20220711-110103"
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_kde" --model_name "20220629-224059" --oblivion
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_dla" --model_name "20220711-122226" --oblivion
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_icnn" --model_name "20220711-130831" --oblivion
python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_pad_ma" --model_name "20230124-082522" --oblivion
done


python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "md_dnn" --model_name "20220629-224059"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "md_at_pgd" --model_name "20220711-103657"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "md_at_ma" --model_name "20220711-110103"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_kde" --model_name "20220629-224059" --oblivion
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_dla" --model_name "20220711-122226" --oblivion
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_dnn_plus" --model_name "20220711-124621" --oblivion
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_icnn" --model_name "20220711-130831" --oblivion
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_pad_ma" --model_name "20230124-082522" --oblivion


# ========================================================================================
# adaptive attacks
# ========================================================================================

for i in {10,20,30}
do
  python -m examples.mimicry_test --trials $i --model "amd_kde" --model_name "20220629-224059"
  python -m examples.mimicry_test --trials $i --model "amd_dla" --model_name "20220711-122226"
  python -m examples.mimicry_test --trials $i --model "amd_dnn_plus" --model_name "20220711-124621"
  python -m examples.mimicry_test --trials $i --model "amd_icnn" --model_name "20220711-130831"
  python -m examples.mimicry_test --trials $i --model "amd_pad_ma" --model_name "20230124-082522"
done
#
for i in {1,2,5}
do
   python -m examples.grosse_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.grosse_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.grosse_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.grosse_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.grosse_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done
for i in {10..120..10}
do
   python -m examples.grosse_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.grosse_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.grosse_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.grosse_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.grosse_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done


for i in {1,2,5}
do
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --kappa 0.2 --random --model "amd_kde" --model_name "20220629-224059"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dla" --model_name "20220711-122226"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_pad_ma" --model_name "20230124-082522"
done
for i in {10..120..10}
do
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --kappa 0.2 --random --model "amd_kde" --model_name "20220629-224059"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dla" --model_name "20220711-122226"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.rfgsm_test --steps $i --step_length 0.02 --random --model "amd_pad_ma" --model_name "20230124-082522"
done

for i in {1,2,5}
do
   python -m examples.bga_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.bga_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.bga_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.bga_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.bga_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done
for i in {10..120..10}
do
   python -m examples.bga_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.bga_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.bga_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.bga_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.bga_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done

for i in {1,2,5}
do
   python -m examples.bca_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.bca_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.bca_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.bca_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.bca_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done
for i in {10..120..10}
do
   python -m examples.bca_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.bca_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.bca_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.bca_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.bca_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done

for i in {1,2,5}
do
   python -m examples.pgdl1_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.pgdl1_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.pgdl1_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.pgdl1_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.pgdl1_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done
for i in {10..120..10}
do
   python -m examples.pgdl1_test --steps $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
   python -m examples.pgdl1_test --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.pgdl1_test --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.pgdl1_test --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.pgdl1_test --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done


python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_dla" --model_name "20220711-122226"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_pad_ma" --model_name "20230124-082522"


python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_dla" --model_name "20220711-122226"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_pad_ma" --model_name "20230124-082522"


# orthogonal
for i in {1,2,5}
do
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --step_length 1. --model "amd_kde" --model_name "20220629-224059"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done
for i in {10..120..10}
do
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_kde" --model_name "20220629-224059"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_dla" --model_name "20220711-122226"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_dnn_plus" --model_name "20220711-124621"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_icnn" --model_name "20220711-130831"
   python -m examples.orthogonal_pgd_test --norm "l1" --project_detector --steps $i --model "amd_pad_ma" --model_name "20230124-082522"
done


python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_kde" --kappa 0.2 --model_name "20220629-224059"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_dla" --model_name "20220711-122226"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.pgd_test --norm "l2" --steps 200 --step_length 0.05 --model "amd_pad_ma" --model_name "20230124-082522"

python -m examples.orthogonal_pgd_test --norm "l2" --project_detector --steps 200 --step_length 0.05 --model "amd_kde" --model_name "20220629-224059"
python -m examples.orthogonal_pgd_test --norm "l2" --project_detector --steps 200 --step_length 0.05 --model "amd_dla" --model_name "20220711-122226"
python -m examples.orthogonal_pgd_test --norm "l2" --project_detector --steps 200 --step_length 0.05 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.orthogonal_pgd_test --norm "l2" --project_detector --steps 200 --step_length 0.05 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.orthogonal_pgd_test --norm "l2" --project_detector --steps 200 --step_length 0.05 --model "amd_pad_ma" --model_name "20230124-082522"


python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_kde" --kappa 0.2 --model_name "20220629-224059"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_dla" --model_name "20220711-122226"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.pgd_test --norm "linf" --steps 500 --step_length 0.002 --model "amd_pad_ma" --model_name "20230124-082522"

python -m examples.orthogonal_pgd_test --norm "linf" --project_detector --project_classifier --steps 500 --step_length 0.002 --model "amd_kde" --model_name "20220629-224059"
python -m examples.orthogonal_pgd_test --norm "linf" --project_detector --project_classifier --steps 500 --step_length 0.002 --model "amd_dla" --model_name "20220711-122226"
python -m examples.orthogonal_pgd_test --norm "linf" --project_detector --project_classifier --steps 500 --step_length 0.002 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.orthogonal_pgd_test --norm "linf" --project_detector --project_classifier --steps 500 --step_length 0.002 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.orthogonal_pgd_test --norm "linf" --project_detector --project_classifier --steps 500 --step_length 0.002 --model "amd_pad_ma" --model_name "20230124-082522"


for i in {1,5};
do
  python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
  python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_dla" --model_name "20220711-122226"
  python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_dnn_plus" --model_name "20220711-124621"
  python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_icnn" --model_name "20220711-130831"
  python -m examples.max_test --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_pad_ma" --model_name "20230124-082522"
done


for i in {1,5};
do
  python -m examples.max_test --orthogonal_v --project_detector --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_kde" --model_name "20220629-224059"
  python -m examples.max_test --orthogonal_v --project_detector --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_dla" --model_name "20220711-122226"
  python -m examples.max_test --orthogonal_v --project_detector --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_dnn_plus" --model_name "20220711-124621"
  python -m examples.max_test --orthogonal_v --project_detector --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_icnn" --model_name "20220711-130831"
  python -m examples.max_test --orthogonal_v --project_detector --steps_l1 50 --steps_linf 500 --step_length_linf 0.002 --steps_l2 200 --step_length_l2 0.05  --steps_max $i --model "amd_pad_ma" --model_name "20230124-082522"
done



python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --kappa 0.2 --model "amd_kde" --model_name "20220629-224059"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_dla" --model_name "20220711-122226"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_dnn_plus" --model_name "20220711-124621"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_icnn" --model_name "20220711-130831"
python -m examples.stepwise_max_test --steps 500 --step_length_linf 0.002 --step_length_l2 0.05 --model "amd_pad_ma" --model_name "20230124-082522"


