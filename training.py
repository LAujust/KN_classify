"""
aujust@mail.ustc.edu.cn
"""

from astrorapid.custom_classifier import create_custom_classifier
import astrorapid
import sys
sys.path.append('/home/Aujust/data/Kilonova/Constraint/')
sys.path.append('/home/Aujust/data/Kilonova/GPR/')
import toolkit as tkk
import pickle

# def get_custom_data(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo, calculate_t0):
    # name_map = {
    #     1:'KN',
    #     2:'Other'
    # }    
    # f_dir = data_dir + name_map[class_num]
    # with open(f_dir+'.pkl','rb') as handle:
    #     data = pickle.load(handle)
    #     handle.close()
    # return data
    
def get_custom_data(class_num, data_dir, save_dir, passbands, known_redshift, nprocesses, redo, calculate_t0):
    'class_num: 1-KN, 2-Other'
    name_map = {
        1:['KN_POSSIS','KN_MOSFiT'],
        2:['SNIa','SNIbc','SNIIn','SNIIP','SNIa_91bg','SLSN']
    }

    out = {}
    
    fns = name_map[class_num]
    for fn in fns:
        f_dir = data_dir + fn + '_raw.pkl'
        with open(f_dir,'rb') as handle:
            data = pickle.load(handle)
            handle.close()
        out.update(data)
    
    # if save_dir:
    #     with open(save_dir,'wb') as handle:
    #         pickle.dump(out,handle)
    #         handle.close()

    return out

create_custom_classifier(get_data_func=get_custom_data,
                         data_dir='/home/Aujust/data/Kilonova/KN_classify/data/Training_set_0908/',
                         class_nums=(1, 2,),
                         class_name_map={1:'Kilonova', 2:'Other'},
                         reread_data=True,
                         contextual_info=('distmu','distsigma','log_prob'),
                         passbands=('wfst_g','wfst_r','wfst_i'),
                         retrain_network=True,
                         train_epochs=50,
                         zcut=False,
                         bcut=False,
                         dropout_rate=0.05,
                         #ignore_classes=(3,),
                         nprocesses=None,
                         nchunks=1000,
                         timestep=0.5,
                         mintime=-2,
                         maxtime=5,
                         otherchange='',
                         init_day_since_trigger=-1,
                         training_set_dir='/home/Aujust/data/Kilonova/KN_classify/data/processed_data_PMD', ## PMD/PM/MD 
                         save_dir='/home/Aujust/data/Kilonova/KN_classify/data/lightcurves_new',
                         fig_dir='/home/Aujust/data/Kilonova/KN_classify/figures1012_ks32_drop01',
                         plot=True)

print()