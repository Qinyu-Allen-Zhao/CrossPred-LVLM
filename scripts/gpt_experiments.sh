python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset SEED_2 \
        --store_path ./output/gpt-4o-2024-05-13/SEED_2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset MME \
        --store_path ./output/gpt-4o-2024-05-13/MME_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset MMBench_CN \
        --store_path ./output/gpt-4o-2024-05-13/MMBench_CN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset MMBench_EN \
        --store_path ./output/gpt-4o-2024-05-13/MMBench_EN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset MMMU \
        --store_path ./output/gpt-4o-2024-05-13/MMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset CMMMU \
        --store_path ./output/gpt-4o-2024-05-13/CMMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset ScienceQA \
        --store_path ./output/gpt-4o-2024-05-13/ScienceQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset CVBench \
        --store_path ./output/gpt-4o-2024-05-13/CVBench_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset DECIMER \
        --store_path ./output/gpt-4o-2024-05-13/DECIMER_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Enrico \
        --store_path ./output/gpt-4o-2024-05-13/Enrico_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset FaceEmotion \
        --store_path ./output/gpt-4o-2024-05-13/FaceEmotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Flickr30k \
        --store_path ./output/gpt-4o-2024-05-13/Flickr30k_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset GQA \
        --store_path ./output/gpt-4o-2024-05-13/GQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset HatefulMemes \
        --store_path ./output/gpt-4o-2024-05-13/HatefulMemes_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset INAT \
        --store_path ./output/gpt-4o-2024-05-13/INAT_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset IRFL \
        --store_path ./output/gpt-4o-2024-05-13/IRFL_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset MemeCaps \
        --store_path ./output/gpt-4o-2024-05-13/MemeCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Memotion \
        --store_path ./output/gpt-4o-2024-05-13/Memotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset MMIMDB \
        --store_path ./output/gpt-4o-2024-05-13/MMIMDB_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset NewYorkerCartoon \
        --store_path ./output/gpt-4o-2024-05-13/NewYorkerCartoon_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset NLVR \
        --store_path ./output/gpt-4o-2024-05-13/NLVR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset NLVR2 \
        --store_path ./output/gpt-4o-2024-05-13/NLVR2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset NoCaps \
        --store_path ./output/gpt-4o-2024-05-13/NoCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset OKVQA \
        --store_path ./output/gpt-4o-2024-05-13/OKVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset OpenPath \
        --store_path ./output/gpt-4o-2024-05-13/OpenPath_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset PathVQA \
        --store_path ./output/gpt-4o-2024-05-13/PathVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Resisc45 \
        --store_path ./output/gpt-4o-2024-05-13/Resisc45_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Screen2Words \
        --store_path ./output/gpt-4o-2024-05-13/Screen2Words_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Slake \
        --store_path ./output/gpt-4o-2024-05-13/Slake_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset UCMerced \
        --store_path ./output/gpt-4o-2024-05-13/UCMerced_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset VCR \
        --store_path ./output/gpt-4o-2024-05-13/VCR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset VisualGenome \
        --store_path ./output/gpt-4o-2024-05-13/VisualGenome_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset VQA \
        --store_path ./output/gpt-4o-2024-05-13/VQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset VQARAD \
        --store_path ./output/gpt-4o-2024-05-13/VQARAD_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset Winoground \
        --store_path ./output/gpt-4o-2024-05-13/Winoground_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-05-13 \
        --dataset POPE \
        --store_path ./output/gpt-4o-2024-05-13/POPE_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset SEED_2 \
        --store_path ./output/gpt-4o-2024-08-06/SEED_2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset MME \
        --store_path ./output/gpt-4o-2024-08-06/MME_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset MMBench_CN \
        --store_path ./output/gpt-4o-2024-08-06/MMBench_CN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset MMBench_EN \
        --store_path ./output/gpt-4o-2024-08-06/MMBench_EN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset MMMU \
        --store_path ./output/gpt-4o-2024-08-06/MMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset CMMMU \
        --store_path ./output/gpt-4o-2024-08-06/CMMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset ScienceQA \
        --store_path ./output/gpt-4o-2024-08-06/ScienceQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset CVBench \
        --store_path ./output/gpt-4o-2024-08-06/CVBench_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset DECIMER \
        --store_path ./output/gpt-4o-2024-08-06/DECIMER_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Enrico \
        --store_path ./output/gpt-4o-2024-08-06/Enrico_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset FaceEmotion \
        --store_path ./output/gpt-4o-2024-08-06/FaceEmotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Flickr30k \
        --store_path ./output/gpt-4o-2024-08-06/Flickr30k_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset GQA \
        --store_path ./output/gpt-4o-2024-08-06/GQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset HatefulMemes \
        --store_path ./output/gpt-4o-2024-08-06/HatefulMemes_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset INAT \
        --store_path ./output/gpt-4o-2024-08-06/INAT_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset IRFL \
        --store_path ./output/gpt-4o-2024-08-06/IRFL_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset MemeCaps \
        --store_path ./output/gpt-4o-2024-08-06/MemeCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Memotion \
        --store_path ./output/gpt-4o-2024-08-06/Memotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset MMIMDB \
        --store_path ./output/gpt-4o-2024-08-06/MMIMDB_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset NewYorkerCartoon \
        --store_path ./output/gpt-4o-2024-08-06/NewYorkerCartoon_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset NLVR \
        --store_path ./output/gpt-4o-2024-08-06/NLVR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset NLVR2 \
        --store_path ./output/gpt-4o-2024-08-06/NLVR2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset NoCaps \
        --store_path ./output/gpt-4o-2024-08-06/NoCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset OKVQA \
        --store_path ./output/gpt-4o-2024-08-06/OKVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset OpenPath \
        --store_path ./output/gpt-4o-2024-08-06/OpenPath_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset PathVQA \
        --store_path ./output/gpt-4o-2024-08-06/PathVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Resisc45 \
        --store_path ./output/gpt-4o-2024-08-06/Resisc45_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Screen2Words \
        --store_path ./output/gpt-4o-2024-08-06/Screen2Words_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Slake \
        --store_path ./output/gpt-4o-2024-08-06/Slake_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset UCMerced \
        --store_path ./output/gpt-4o-2024-08-06/UCMerced_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset VCR \
        --store_path ./output/gpt-4o-2024-08-06/VCR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset VisualGenome \
        --store_path ./output/gpt-4o-2024-08-06/VisualGenome_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset VQA \
        --store_path ./output/gpt-4o-2024-08-06/VQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset VQARAD \
        --store_path ./output/gpt-4o-2024-08-06/VQARAD_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset Winoground \
        --store_path ./output/gpt-4o-2024-08-06/Winoground_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name GPT4 \
        --model_path gpt-4o-2024-08-06 \
        --dataset POPE \
        --store_path ./output/gpt-4o-2024-08-06/POPE_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset SEED_2 \
        --store_path ./output/gpt-4o-mini-2024-07-18/SEED_2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset MME \
        --store_path ./output/gpt-4o-mini-2024-07-18/MME_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset MMBench_CN \
        --store_path ./output/gpt-4o-mini-2024-07-18/MMBench_CN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset MMBench_EN \
        --store_path ./output/gpt-4o-mini-2024-07-18/MMBench_EN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset MMMU \
        --store_path ./output/gpt-4o-mini-2024-07-18/MMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset CMMMU \
        --store_path ./output/gpt-4o-mini-2024-07-18/CMMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset ScienceQA \
        --store_path ./output/gpt-4o-mini-2024-07-18/ScienceQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset CVBench \
        --store_path ./output/gpt-4o-mini-2024-07-18/CVBench_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset DECIMER \
        --store_path ./output/gpt-4o-mini-2024-07-18/DECIMER_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Enrico \
        --store_path ./output/gpt-4o-mini-2024-07-18/Enrico_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset FaceEmotion \
        --store_path ./output/gpt-4o-mini-2024-07-18/FaceEmotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Flickr30k \
        --store_path ./output/gpt-4o-mini-2024-07-18/Flickr30k_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset GQA \
        --store_path ./output/gpt-4o-mini-2024-07-18/GQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset HatefulMemes \
        --store_path ./output/gpt-4o-mini-2024-07-18/HatefulMemes_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset INAT \
        --store_path ./output/gpt-4o-mini-2024-07-18/INAT_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset IRFL \
        --store_path ./output/gpt-4o-mini-2024-07-18/IRFL_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset MemeCaps \
        --store_path ./output/gpt-4o-mini-2024-07-18/MemeCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Memotion \
        --store_path ./output/gpt-4o-mini-2024-07-18/Memotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset MMIMDB \
        --store_path ./output/gpt-4o-mini-2024-07-18/MMIMDB_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset NewYorkerCartoon \
        --store_path ./output/gpt-4o-mini-2024-07-18/NewYorkerCartoon_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset NLVR \
        --store_path ./output/gpt-4o-mini-2024-07-18/NLVR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset NLVR2 \
        --store_path ./output/gpt-4o-mini-2024-07-18/NLVR2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset NoCaps \
        --store_path ./output/gpt-4o-mini-2024-07-18/NoCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset OKVQA \
        --store_path ./output/gpt-4o-mini-2024-07-18/OKVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset OpenPath \
        --store_path ./output/gpt-4o-mini-2024-07-18/OpenPath_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset PathVQA \
        --store_path ./output/gpt-4o-mini-2024-07-18/PathVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Resisc45 \
        --store_path ./output/gpt-4o-mini-2024-07-18/Resisc45_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Screen2Words \
        --store_path ./output/gpt-4o-mini-2024-07-18/Screen2Words_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Slake \
        --store_path ./output/gpt-4o-mini-2024-07-18/Slake_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset UCMerced \
        --store_path ./output/gpt-4o-mini-2024-07-18/UCMerced_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset VCR \
        --store_path ./output/gpt-4o-mini-2024-07-18/VCR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset VisualGenome \
        --store_path ./output/gpt-4o-mini-2024-07-18/VisualGenome_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset VQA \
        --store_path ./output/gpt-4o-mini-2024-07-18/VQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset VQARAD \
        --store_path ./output/gpt-4o-mini-2024-07-18/VQARAD_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset Winoground \
        --store_path ./output/gpt-4o-mini-2024-07-18/Winoground_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name GPT4 \
        --model_path gpt-4o-mini-2024-07-18 \
        --dataset POPE \
        --store_path ./output/gpt-4o-mini-2024-07-18/POPE_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset SEED_2 \
        --store_path ./output/gpt-4-turbo-2024-04-09/SEED_2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset MME \
        --store_path ./output/gpt-4-turbo-2024-04-09/MME_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset MMBench_CN \
        --store_path ./output/gpt-4-turbo-2024-04-09/MMBench_CN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset MMBench_EN \
        --store_path ./output/gpt-4-turbo-2024-04-09/MMBench_EN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset MMMU \
        --store_path ./output/gpt-4-turbo-2024-04-09/MMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset CMMMU \
        --store_path ./output/gpt-4-turbo-2024-04-09/CMMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset ScienceQA \
        --store_path ./output/gpt-4-turbo-2024-04-09/ScienceQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset CVBench \
        --store_path ./output/gpt-4-turbo-2024-04-09/CVBench_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset DECIMER \
        --store_path ./output/gpt-4-turbo-2024-04-09/DECIMER_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Enrico \
        --store_path ./output/gpt-4-turbo-2024-04-09/Enrico_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset FaceEmotion \
        --store_path ./output/gpt-4-turbo-2024-04-09/FaceEmotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Flickr30k \
        --store_path ./output/gpt-4-turbo-2024-04-09/Flickr30k_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset GQA \
        --store_path ./output/gpt-4-turbo-2024-04-09/GQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset HatefulMemes \
        --store_path ./output/gpt-4-turbo-2024-04-09/HatefulMemes_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset INAT \
        --store_path ./output/gpt-4-turbo-2024-04-09/INAT_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset IRFL \
        --store_path ./output/gpt-4-turbo-2024-04-09/IRFL_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset MemeCaps \
        --store_path ./output/gpt-4-turbo-2024-04-09/MemeCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Memotion \
        --store_path ./output/gpt-4-turbo-2024-04-09/Memotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset MMIMDB \
        --store_path ./output/gpt-4-turbo-2024-04-09/MMIMDB_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset NewYorkerCartoon \
        --store_path ./output/gpt-4-turbo-2024-04-09/NewYorkerCartoon_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset NLVR \
        --store_path ./output/gpt-4-turbo-2024-04-09/NLVR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset NLVR2 \
        --store_path ./output/gpt-4-turbo-2024-04-09/NLVR2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset NoCaps \
        --store_path ./output/gpt-4-turbo-2024-04-09/NoCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset OKVQA \
        --store_path ./output/gpt-4-turbo-2024-04-09/OKVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset OpenPath \
        --store_path ./output/gpt-4-turbo-2024-04-09/OpenPath_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset PathVQA \
        --store_path ./output/gpt-4-turbo-2024-04-09/PathVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Resisc45 \
        --store_path ./output/gpt-4-turbo-2024-04-09/Resisc45_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Screen2Words \
        --store_path ./output/gpt-4-turbo-2024-04-09/Screen2Words_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Slake \
        --store_path ./output/gpt-4-turbo-2024-04-09/Slake_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset UCMerced \
        --store_path ./output/gpt-4-turbo-2024-04-09/UCMerced_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset VCR \
        --store_path ./output/gpt-4-turbo-2024-04-09/VCR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset VisualGenome \
        --store_path ./output/gpt-4-turbo-2024-04-09/VisualGenome_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset VQA \
        --store_path ./output/gpt-4-turbo-2024-04-09/VQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset VQARAD \
        --store_path ./output/gpt-4-turbo-2024-04-09/VQARAD_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset Winoground \
        --store_path ./output/gpt-4-turbo-2024-04-09/Winoground_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name GPT4 \
        --model_path gpt-4-turbo-2024-04-09 \
        --dataset POPE \
        --store_path ./output/gpt-4-turbo-2024-04-09/POPE_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset SEED_2 \
        --store_path ./output/gemini-1.5-pro/SEED_2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset MME \
        --store_path ./output/gemini-1.5-pro/MME_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset MMBench_CN \
        --store_path ./output/gemini-1.5-pro/MMBench_CN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset MMBench_EN \
        --store_path ./output/gemini-1.5-pro/MMBench_EN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset MMMU \
        --store_path ./output/gemini-1.5-pro/MMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset CMMMU \
        --store_path ./output/gemini-1.5-pro/CMMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset ScienceQA \
        --store_path ./output/gemini-1.5-pro/ScienceQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset CVBench \
        --store_path ./output/gemini-1.5-pro/CVBench_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset DECIMER \
        --store_path ./output/gemini-1.5-pro/DECIMER_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Enrico \
        --store_path ./output/gemini-1.5-pro/Enrico_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset FaceEmotion \
        --store_path ./output/gemini-1.5-pro/FaceEmotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Flickr30k \
        --store_path ./output/gemini-1.5-pro/Flickr30k_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset GQA \
        --store_path ./output/gemini-1.5-pro/GQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset HatefulMemes \
        --store_path ./output/gemini-1.5-pro/HatefulMemes_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset INAT \
        --store_path ./output/gemini-1.5-pro/INAT_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset IRFL \
        --store_path ./output/gemini-1.5-pro/IRFL_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset MemeCaps \
        --store_path ./output/gemini-1.5-pro/MemeCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Memotion \
        --store_path ./output/gemini-1.5-pro/Memotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset MMIMDB \
        --store_path ./output/gemini-1.5-pro/MMIMDB_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset NewYorkerCartoon \
        --store_path ./output/gemini-1.5-pro/NewYorkerCartoon_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset NLVR \
        --store_path ./output/gemini-1.5-pro/NLVR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset NLVR2 \
        --store_path ./output/gemini-1.5-pro/NLVR2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset NoCaps \
        --store_path ./output/gemini-1.5-pro/NoCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset OKVQA \
        --store_path ./output/gemini-1.5-pro/OKVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset OpenPath \
        --store_path ./output/gemini-1.5-pro/OpenPath_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset PathVQA \
        --store_path ./output/gemini-1.5-pro/PathVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Resisc45 \
        --store_path ./output/gemini-1.5-pro/Resisc45_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Screen2Words \
        --store_path ./output/gemini-1.5-pro/Screen2Words_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Slake \
        --store_path ./output/gemini-1.5-pro/Slake_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset UCMerced \
        --store_path ./output/gemini-1.5-pro/UCMerced_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset VCR \
        --store_path ./output/gemini-1.5-pro/VCR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset VisualGenome \
        --store_path ./output/gemini-1.5-pro/VisualGenome_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset VQA \
        --store_path ./output/gemini-1.5-pro/VQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset VQARAD \
        --store_path ./output/gemini-1.5-pro/VQARAD_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset Winoground \
        --store_path ./output/gemini-1.5-pro/Winoground_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name Gemini \
        --model_path gemini-1.5-pro \
        --dataset POPE \
        --store_path ./output/gemini-1.5-pro/POPE_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset SEED_2 \
        --store_path ./output/gemini-1.5-flash/SEED_2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset MME \
        --store_path ./output/gemini-1.5-flash/MME_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset MMBench_CN \
        --store_path ./output/gemini-1.5-flash/MMBench_CN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset MMBench_EN \
        --store_path ./output/gemini-1.5-flash/MMBench_EN_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset MMMU \
        --store_path ./output/gemini-1.5-flash/MMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset CMMMU \
        --store_path ./output/gemini-1.5-flash/CMMMU_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset ScienceQA \
        --store_path ./output/gemini-1.5-flash/ScienceQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset CVBench \
        --store_path ./output/gemini-1.5-flash/CVBench_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset DECIMER \
        --store_path ./output/gemini-1.5-flash/DECIMER_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Enrico \
        --store_path ./output/gemini-1.5-flash/Enrico_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset FaceEmotion \
        --store_path ./output/gemini-1.5-flash/FaceEmotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Flickr30k \
        --store_path ./output/gemini-1.5-flash/Flickr30k_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset GQA \
        --store_path ./output/gemini-1.5-flash/GQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset HatefulMemes \
        --store_path ./output/gemini-1.5-flash/HatefulMemes_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset INAT \
        --store_path ./output/gemini-1.5-flash/INAT_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset IRFL \
        --store_path ./output/gemini-1.5-flash/IRFL_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset MemeCaps \
        --store_path ./output/gemini-1.5-flash/MemeCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Memotion \
        --store_path ./output/gemini-1.5-flash/Memotion_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset MMIMDB \
        --store_path ./output/gemini-1.5-flash/MMIMDB_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset NewYorkerCartoon \
        --store_path ./output/gemini-1.5-flash/NewYorkerCartoon_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset NLVR \
        --store_path ./output/gemini-1.5-flash/NLVR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset NLVR2 \
        --store_path ./output/gemini-1.5-flash/NLVR2_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset NoCaps \
        --store_path ./output/gemini-1.5-flash/NoCaps_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset OKVQA \
        --store_path ./output/gemini-1.5-flash/OKVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset OpenPath \
        --store_path ./output/gemini-1.5-flash/OpenPath_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset PathVQA \
        --store_path ./output/gemini-1.5-flash/PathVQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Resisc45 \
        --store_path ./output/gemini-1.5-flash/Resisc45_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Screen2Words \
        --store_path ./output/gemini-1.5-flash/Screen2Words_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Slake \
        --store_path ./output/gemini-1.5-flash/Slake_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset UCMerced \
        --store_path ./output/gemini-1.5-flash/UCMerced_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset VCR \
        --store_path ./output/gemini-1.5-flash/VCR_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset VisualGenome \
        --store_path ./output/gemini-1.5-flash/VisualGenome_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset VQA \
        --store_path ./output/gemini-1.5-flash/VQA_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset VQARAD \
        --store_path ./output/gemini-1.5-flash/VQARAD_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset Winoground \
        --store_path ./output/gemini-1.5-flash/Winoground_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name Gemini \
        --model_path gemini-1.5-flash \
        --dataset POPE \
        --store_path ./output/gemini-1.5-flash/POPE_HS/ \
        --num_chunks 1 \
        --chunk_idx 0 \
        --temperature 0.0 \
        --num_beams 1 \
        --only_ans &
wait

