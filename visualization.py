import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
#%%
def Compute_Concentraion_matrix(w):
    """ Compute and plot a concentration matrix
    Args:
        w : (float array, [D, num_label]) last fully connected layers's weights
    """
    w = w/np.linalg.norm(w,axis=0,keepdims=True)
    C = w.T.dot(w)
    C = (C-np.min(C,1,keepdims=True))/(np.max(C,1,keepdims=True)-np.min(C,1,keepdims=True))
    K = C.shape[0]
    
    plt.rcParams["figure.figsize"] = (10,10)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 15})
    
    plt.xticks(np.arange(0,K), np.arange(0,K))
    plt.yticks(np.arange(0,K), np.arange(0,K))
    plt.imshow(C)
    
    plt.colorbar(fraction = 0.045)

#%% Visualize concentration parameter
home_path = os.path.dirname(os.path.abspath(__file__))
params = sio.loadmat(home_path + '/pre_trained/Lenet5.mat')
w = params['Teacher/fc2/weights']
C = Compute_Concentraion_matrix(w)

#%% Visualize generated DI samples
data = sio.loadmat(home_path + '/DI/DI-40.mat')
for i, l in zip(data['train_images'], data['train_labels']):
    first, second = sorted(np.vstack([np.arange(10),l]).T, key=lambda l: l[1], reverse = True)[:2]
    
    image = np.pad(cv2.resize(i,(200,200)).astype(np.uint8),[[24,0],[0,0]], 'constant')
    cv2.putText(image,'\'%d\':%.2f, \'%d\':%.2f'%(first[0], first[1], second[0], second[1] ),
                (0,24), cv2.FONT_HERSHEY_SIMPLEX, .75,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('',image)
    key = cv2.waitKey(0) & 0xff
    if key == ord('q'):
        break
cv2.destroyAllWindows()
