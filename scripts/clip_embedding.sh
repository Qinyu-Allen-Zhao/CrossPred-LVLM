#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset SEED_2 \
        --store_path ./output/CLIP_embed/SEED_2_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset MME \
        --store_path ./output/CLIP_embed/MME_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset MMBench_CN \
        --store_path ./output/CLIP_embed/MMBench_CN_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset MMBench_EN \
        --store_path ./output/CLIP_embed/MMBench_EN_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset MMMU \
        --store_path ./output/CLIP_embed/MMMU_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset CMMMU \
        --store_path ./output/CLIP_embed/CMMMU_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset ScienceQA \
        --store_path ./output/CLIP_embed/ScienceQA_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset CVBench \
        --store_path ./output/CLIP_embed/CVBench_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset DECIMER \
        --store_path ./output/CLIP_embed/DECIMER_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Enrico \
        --store_path ./output/CLIP_embed/Enrico_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset FaceEmotion \
        --store_path ./output/CLIP_embed/FaceEmotion_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Flickr30k \
        --store_path ./output/CLIP_embed/Flickr30k_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset GQA \
        --store_path ./output/CLIP_embed/GQA_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset HatefulMemes \
        --store_path ./output/CLIP_embed/HatefulMemes_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset INAT \
        --store_path ./output/CLIP_embed/INAT_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset IRFL \
        --store_path ./output/CLIP_embed/IRFL_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset MemeCaps \
        --store_path ./output/CLIP_embed/MemeCaps_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Memotion \
        --store_path ./output/CLIP_embed/Memotion_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset MMIMDB \
        --store_path ./output/CLIP_embed/MMIMDB_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset NewYorkerCartoon \
        --store_path ./output/CLIP_embed/NewYorkerCartoon_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset NLVR \
        --store_path ./output/CLIP_embed/NLVR_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset NLVR2 \
        --store_path ./output/CLIP_embed/NLVR2_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset NoCaps \
        --store_path ./output/CLIP_embed/NoCaps_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset OKVQA \
        --store_path ./output/CLIP_embed/OKVQA_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset OpenPath \
        --store_path ./output/CLIP_embed/OpenPath_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset PathVQA \
        --store_path ./output/CLIP_embed/PathVQA_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Resisc45 \
        --store_path ./output/CLIP_embed/Resisc45_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Screen2Words \
        --store_path ./output/CLIP_embed/Screen2Words_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Slake \
        --store_path ./output/CLIP_embed/Slake_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset UCMerced \
        --store_path ./output/CLIP_embed/UCMerced_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset VCR \
        --store_path ./output/CLIP_embed/VCR_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset VisualGenome \
        --store_path ./output/CLIP_embed/VisualGenome_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset VQA \
        --store_path ./output/CLIP_embed/VQA_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset VQARAD \
        --store_path ./output/CLIP_embed/VQARAD_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 100 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset Winoground \
        --store_path ./output/CLIP_embed/Winoground_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m evaluate_lvlm \
        --num_samples 900 \
        --model_name CLIP \
        --model_path ViT-B/32 \
        --dataset POPE \
        --store_path ./output/CLIP_embed/POPE_HS/ \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1

done

wait

