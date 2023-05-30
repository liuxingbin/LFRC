CUDA_VISIBLE_DEVICES=0 python eval_cifar.py \
        --batch-size=256 \
        --type=best \
        --data=cifar10 \
        --data-dir=~/datasets \
        --epsilon=8 \
        --model=ResNet18 \
        --out-dir=~/resnet_cifar10_at
