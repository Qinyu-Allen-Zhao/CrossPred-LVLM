for i in {1..10}
do
    echo "Round $i"
    for dim in 5 10 15 20 50 100
    do
        python run_pmf.py --random_seed $i --percent_test 20 --subset OneMat --dim $dim --eval_train
    done
done
