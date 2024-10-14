for i in {1..10}
do
    echo "Round $i"
    python active_eval.py --method random --seed $i
    wait
    python active_eval.py --method active --seed $i
    wait
    python active_eval.py --method worst --seed $i
    wait
done