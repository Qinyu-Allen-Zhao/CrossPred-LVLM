percent_test=20

for i in {1..10}
do
    echo "Percent Test: $percent_test | Round $i"
    python cold_start.py --random_seed $i --percent_test $percent_test --method mean
    python cold_start.py --random_seed $i --percent_test $percent_test --method pmf
    # python cold_start.py --random_seed $i --percent_test $percent_test --method cpmf --dataset_profile random
    # python cold_start.py --random_seed $i --percent_test $percent_test --method cpmf --dataset_profile des_cluster
    # python cold_start.py --random_seed $i --percent_test $percent_test --method cpmf --dataset_profile clip_cluster
    python cold_start.py --random_seed $i --percent_test $percent_test --method cpmf --dataset_profile hs_cluster
    # python cold_start.py --random_seed $i --percent_test $percent_test --method cpmf --dataset_profile gt_cluster
    python cold_start.py --random_seed $i --percent_test $percent_test --method cpmf --gt_profile
    wait
done
