for percent_test in 20 40 60 80 90 95
do
    for i in {1..10}
    do
        echo "Percent Test: $percent_test | Round $i"
        python run_ptf.py --random_seed $i --percent_test $percent_test --method ptf
        python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf --gt_profile
        python run_ptf.py --random_seed $i --percent_test $percent_test --method cptf
        python run_ptf.py --random_seed $i --percent_test $percent_test --method bptf
        python run_ptf.py --random_seed $i --percent_test $percent_test --method bcptf --gt_profile
        python run_ptf.py --random_seed $i --percent_test $percent_test --method bcptf
        wait
    done
done