# Set test percentage
for percent_test in 20 40 60 80 90 95
do
    echo "Percent Test: $percent_test"
    for i in {1..10}
    do
        echo "Percent Test: $percent_test | Round $i"
        python run_pmf.py --random_seed $i --percent_test $percent_test --subset OneMat
        wait

    done
done