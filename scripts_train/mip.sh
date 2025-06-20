for SCENE in bonsai counter garden kitchen room stump bicycle
do
  python render.py --source_path  /home/ubuntu/Documents/dataset/mipnerf360/$SCENE --model_path output/mip360/$SCENE --iteration 10000
done

# for SCENE in bonsai counter garden kitchen room stump bicycle
# do
#   CUDA_VISIBLE_DEVICES=0 
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   python train.py --source_path  /home/ubuntu/Documents/dataset/mipnerf360/$SCENE --model_path output/mip360/$SCENE --eval --n_views 24 --D 0.1 --W 0.25 --N 0.1
# done