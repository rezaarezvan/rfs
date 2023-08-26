#!/bin/bash
dirs=("CNN" "GAN" "GPT" "RNN" "ViT" "backprop" "transformers")
> requirements.txt

for dir in "${dirs[@]}"; do
    pipreqs $dir --force
    cat $dir/requirements.txt >> requirements.txt
    rm $dir/requirements.txt
done

sort -u requirements.txt -o requirements.txt
