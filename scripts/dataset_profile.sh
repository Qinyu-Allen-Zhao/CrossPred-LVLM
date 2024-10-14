percent_test=90

for i in {1..10}
do
    echo "Percent Test: $percent_test | Round $i"
    python run_ptf.py --random_seed $i --percent_test $percent_test --method ptf
    python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --dataset_profile random
    # python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --dataset_profile des_cluster
    # python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --dataset_profile clip_cluster
    python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --dataset_profile hs_cluster
    python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --dataset_profile gt_cluster
    python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --gt_profile
    wait
done
