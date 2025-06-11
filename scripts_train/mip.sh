for SCENE in bonsai counter garden kitchen room stump bicycle
do
  #CUDA_VISIBLE_DEVICES=0 
  #export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python render.py --source_path  /home/ubuntu/Documents/dataset/mipnerf360/$SCENE --model_path output/mip360/$SCENE --iteration 10000
done