### This code is built for verifying implemented all the things at once.
import subprocess, os
home_path = os.path.dirname(os.path.abspath(__file__))
###Train teacher network and save parameter named by architecture name.
for i in range(1):
    subprocess.call('python %s/train_w_distill.py '%home_path
                    +'--train_dir=%s/MNIST/Teacher/tch%d '%(home_path, i)
                    +'--Distillation=None '
                    +'--main_scope=Teacher',
                    shell=True)

for i in range(1):
    # Train student by original dataset
    subprocess.call('python %s/train_w_distill.py '%home_path
                    +'--train_dir=%s/MNIST/Student/std%d '%(home_path, i)
                    +'--Distillation=None',
                    shell=True)
print ('Training Done')

### Train student network transferred knowledge by "Zero-shot Knowledge Distillation" on the various sample rates.
for R in [1, 5, 10, 25, 40]: # Run the code for various number of DI samples
    for i in range(1):     # Run the code a few times for reducing variance.
        # Make data impression samples
        subprocess.call('python %s/Data_Impressions.py '%home_path
                        +'--Rate=%d '%R,
                        shell=True)
        # Train student by data impression samples
        subprocess.call('python %s/train_w_distill.py '%home_path
                        +'--train_dir=%s/MNIST/ZSKD%d/zskd%d_%d '%(home_path, R,R,i)
                        +'--Distillation=ZSKD-%d'%R,
                        shell=True)
print ('Training Done')

### Train student network transferred knowledge by "Soft logits" on the various sample rates.
for R in [1, 5, 10, 25, 40]: # Run the code for various number of samples
    for i in range(1): # Run the code a few times for reducing variance.
        # Train student by sampled original dataset
        subprocess.call('python %s/train_w_distill.py '%home_path
                        +'--train_dir=%s/MNIST/Soft_logits%d/sl%d_%d '%(home_path,R,R,i)
                        +'--Distillation=Soft_logits-%d'%R,
                        shell=True)

