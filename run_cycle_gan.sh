python train.py --dataroot /skydive/Datasets/paired_cyclegan/extracted_eyes --name eyes_cyclegan --model cycle_gan --checkpoints_dir=/skydive/Research-Result/Tavi/cyclegan_extracted_eyes1 --rotation_angle=10 --preprocess=rotate_resize --no_flip

# for cnvrg experiment
# python train.py --dataroot /data/extracted_eyes/ --name eyes_cyclegan --model cycle_gan --checkpoints_dir=output --rotation_angle=10 --preprocess=rotate_resize --no_flip
