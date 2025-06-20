# for SCENE in fern flower fortress horns leaves orchids room trex
# do
#   #CUDA_VISIBLE_DEVICES=0
#   #export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  
#   python render.py --source_path /home/ubuntu/Documents/dataset/nerf_llff_data/$SCENE --model_path output/llff/$SCENE --iteration 10000
# done

for SCENE in fern flower fortress horns leaves orchids room trex
do
  CUDA_VISIBLE_DEVICES=0
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  
  python train.py --source_path /home/ubuntu/Documents/dataset/nerf_llff_data/$SCENE --model_path output/llff/$SCENE --eval --n_views 9 --sample_pseudo_interval 1 --D 1 --W 0.5 --N 1
done