import numpy as np

depth_wl=[1,2,3,4]
for k in depth_wl:

    for save_name in [f'validation_train_{k}',f'train_train_{k}']:

        save_name=f'train_train_{k}'
        loaded=np.load('ComputeKernels\\gram_saves\\'+save_name+'.npz')

        dic={}
        for key in loaded.keys():
            dic[key]=loaded[key].astype(np.int8)

        np.savez_compressed('ComputeKernels\\gram_saves\\'+save_name+'.npz',**dic)
