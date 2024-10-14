for i in {1..10}
do
    python separate_modeling.py --random_seed $i --percent_test 20 --method pmf
done

for i in {1..10}
do
    python separate_modeling.py --random_seed $i --percent_test 90 --method pmf
done