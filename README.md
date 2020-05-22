1. this repo is aboutleveraging the pretrained model Bert to do nmt task. It inculdes:

(1) directly use Bert to get the embedding vector (Bert frozen)

(2) training Bert and transformer together in NMT task to finetune Bert

(3) use the finetuned Bert to get the embedding vector (Bert frozen)


2. the code is based on https://github.com/quanpn90/NMTGMinor.git and  https://github.com/huggingface/transformers
