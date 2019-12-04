
import os


# models_file = "models/mb3-ssd-lite-Epoch-195-Loss-3.0972117355891635.pth "

models_file = "models/mb3-ssd-lite-Epoch-199-Loss-2.539327791758946.pth"
resumemodel = " --resume "+ models_file
datasets = '/home/grobot/mywork/cocodata/fuzi'
num_epochs = 200
batch_size = 10
os.system("python3 train_ssd.py " + resumemodel + " --dataset_type voc --datasets "+datasets+' --net mb3-ssd-lite --scheduler cosine --lr 0.01 --t_max 100 --validation_epochs 5 --num_epochs '+str(num_epochs)+ ' --base_net_lr 0.001 --batch_size ' +str(batch_size))

