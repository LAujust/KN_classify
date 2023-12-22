import KN_classify as KNC
import numpy as np
import toolkit as tkk
import healpy as hp
import pandas as pd
import pickle
import os
import sys
sys.path.append('/home/Aujust/data/Kilonova/WFST/')
from limited_mag import getlim
from ligo.skymap.io.fits import read_sky_map
from astropy.time import Time
import simsurvey
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import markdown
global wfst_fields

event_list = ['S230604z','S230615az','S230627c','S230705bd','S230706al','S230729p']

gw_trigger_map = {}
random_start = '2024-01-01T00:00:00'
t = Time(random_start, format='isot', scale='utc')
random_start_mjd = t.mjd

test_num = 50
test_index_base = [0,200,400,600,800] 
for i,base in enumerate(test_index_base):
    seed_num = i
    np.random.seed(seed_num)
    random_time = np.random.uniform(0, 365, test_num)
    test_index = np.random.choice(200, test_num, replace=False)

    test_index_each = test_index+base
    for ttime,index in zip(random_time, test_index_each):
        gw_trigger_map[index] = ttime + random_start_mjd
print()

def get_basics(events):
    skymaps = []
    plans = []
    gw_triggers = []
    for event_name in events:
        mjd_start = tkk.event_mjd(event_name)
        plan_dir = '/home/Aujust/data/Kilonova/Constraint/plans/{}_wfst_plan.pkl'.format(event_name)
        with open(plan_dir,'rb') as handle:
            plan = pickle.load(handle)
        handle.close()
        skymap_file = '/home/Aujust/data/Kilonova/Constraint/Skymaps/BNS/bayestar_{}.fits'.format(event_name)
        prob, distmu,distsigma,distnorm = hp.fitsfunc.read_map(skymap_file,field=[0,1,2,3])
        skymap = {
                    'prob':prob,
                    'distmu':distmu,
                    'distsigma':distsigma,
                    }
        
        skymaps.append(skymap)
        plans.append(plan)
        gw_triggers.append(mjd_start)
    fields_file = np.loadtxt('/home/Aujust/data/Kilonova/WFST/WFST_square.tess')
    return skymaps,plans,gw_triggers,fields_file


def read_skymap_fits(file_dir,event_id=None):
    #m, meta = read_sky_map(file_dir,distances=True)
    #prob, distmu,distsigma = m[0], m[1], m[2]
    prob,diststd, distmu,distsigma = hp.fitsfunc.read_map(file_dir,field=[0,1,2,3])
    skymap = {
                'prob':prob,
                'distmu':distmu,
                'distsigma':distsigma,
                'event_id':event_id
                    }
    return skymap

def f_filter(f):
    if f[0] in ['.']:
        return False
    else:
        return True
    
def lim_mag(survey_file):
    bands_index = {'g':1,'r':2,'i':3}
    survey_file['maglim'] = [getlim(int(survey_file['exposure_time'].iloc[i]),bgsky=22.0,n_frame=1,airmass=survey_file['airmass'].iloc[i],sig=5)[0][bands_index[survey_file['filt'].iloc[i]]] for i in range(len(survey_file.index))]
    #survey_file['maglim'] = [default_maglim[survey_file.loc[i,'filt']]+1.25*np.log10(survey_file.loc[i,'exposure_time']/30) for i in range(len(survey_file.index))]
    return survey_file
    

def read_datas(survey_dir,skymap_dir):
    survey_parent_fnames = os.listdir(survey_dir)
    survey_parent_fnames = list(filter(f_filter,survey_parent_fnames))
    skymap_fnames = os.listdir(skymap_dir)
    fields_file = np.loadtxt('/home/Aujust/data/Kilonova/WFST/WFST_square.tess')
    wfst_fields = dict()
    wfst_fields['field_id'] = fields_file[:,0].astype(int)
    wfst_fields['ra'] = fields_file[:,1]
    wfst_fields['dec'] = fields_file[:,2]
    
    output = []
    for survey_parent_fname in survey_parent_fnames:
        event_id = survey_parent_fname[3:]
        file_elements = os.listdir('{}/{}'.format(survey_dir,survey_parent_fname))
        skymap_fname = '{}.fits'.format(event_id)
        if 'final' in file_elements and skymap_fname in skymap_fnames:
            print('Pre-processing {}'.format(skymap_fname))
            skymap= read_skymap_fits('{}/{}'.format(skymap_dir,skymap_fname),event_id=event_id)
            #gps_time = meta['gps_time']
            #t = Time(gps_time, format='gps')
            #gw_trigger = t.mjd
            gw_trigger = gw_trigger_map[int(event_id)]
        
            survey_list = os.listdir('{}/{}/final/'.format(survey_dir,survey_parent_fname))
            survey = None
            for i,fname in enumerate(survey_list):
                survey_file = pd.read_csv('{}/{}/final/{}'.format(survey_dir,survey_parent_fname,fname))
                if survey is not None:
                    survey_file = pd.read_csv('{}/{}/final/{}'.format(survey_dir,survey_parent_fname,fname))
                    survey = survey.append(survey_file,ignore_index=True)
                else:
                    survey = survey_file
            output.append((event_id,skymap,survey,gw_trigger,wfst_fields))
        
    return output
            
        
    
       
def get_basic_ms(survey_dir,skymap_dir,plan_save_dir=None,use_plan=False,plan_dir=None):
    data = read_datas(survey_dir,skymap_dir)
    event_ids = [item[0] for item in data]
    skymaps = [item[1] for item in data]
    surveys = [item[2] for item in data]
    gw_triggers = [item[3] for item in data]
    
    wfst_fields = 0
    
    if use_plan:
        plans = []
        print('Using processed plans.')
        for event_id in event_ids:
            with open('{}/{}.pkl'.format(plan_dir,event_id),'rb') as handle:
                p = pickle.load(handle)
                handle.close()
            plans.append(p)
    else:
        print('Multiprocessing lim-mag calculation and convert to simsurvey.plan format')
        with Pool(20) as p:
            out = p.map(process_plan_multi,data)
            p.close()
            p.join()
            
        #output file
        plans = [item[2] for item in out]
    
    if plan_save_dir is not None:
        for i,p in enumerate(plans):
            with open('{}/{}.pkl'.format(plan_save_dir,event_ids[i]),'wb') as handle:
                pickle.dump(p,handle)
                handle.close()

    return event_ids, skymaps, plans, gw_triggers

def process_plan_multi(data_i):
    #====Convert to simsurvey.Plan format===#
    event_id, skymap, survey, gw_trigger, wfst_fields = data_i
    print('Processing {}'.format(event_id))
    survey['band'] = ['wfst_'+survey.loc[i,'filt'] for i in range(len(survey['filt'].index))]
    survey = lim_mag(survey)
    survey['time'] = survey['observ_time']
    survey['field'] = survey['field_id']
    survey = survey.loc[:,['time','field','band','maglim']]
                
    obs = {'time': [], 'field': [], 'band': [], 'maglim': [], 'skynoise': [], 'comment': [], 'zp': []}

    for k in survey.keys():
        obs[k] = survey[k]

        #Defult zp=26.
    obs['zp'] = [26 for i in range(len(survey.index))]
    obs['comment'] = ['' for i in range(len(survey.index))]               
    obs['skynoise'] = 10**(-0.4 * (np.array(obs['maglim']) - 26)) / 5
                
    plan = simsurvey.SurveyPlan(time=obs['time'],
                                band=obs['band'],
                                skynoise=obs['skynoise'],
                                obs_field=obs['field'],
                                zp=obs['zp'],
                                comment=obs['comment'],
                                fields=wfst_fields,
                                #ccds=ccds,
                                width=2.6,
                                height=2.6
                                )

    # if plan_save_dir is not None:
    #     print('Saving plan for skymap_index:{}'.format(survey_parent_fname[3:]))
    #     with open('{}/{}.pkl'.format(plan_save_dir,survey_parent_fname[3:]),'wb') as f:
    #         pickle.dump(plan,f)
    #         f.close()
                
    return (event_id,skymap,plan,gw_trigger)
        


if __name__ == '__main__':
    # id = 555
    # skymap= read_skymap_fits('/home/Aujust/data/Kilonova/Constraint/Skymaps/bns_astro/bns_skymap/{}.fits'.format(id),event_id=id)
    # fig = hp.mollview(skymap['prob'], title=str(id))
    # plt.savefig('/home/Aujust/data/Kilonova/KN_classify/figures/skymap_testshow.jpg')
    event_ids, skymaps, plans, gw_triggers = get_basic_ms('/home/Aujust/data/Kilonova/WFST/ms_event',
                 '/home/Aujust/data/Kilonova/Constraint/Skymaps/bns_astro/bns_skymap',
                 use_plan=True,
                 plan_dir='/home/Aujust/data/Kilonova/Constraint/plans/MS')
    print()
    fields = np.loadtxt('/home/Aujust/data/Kilonova/WFST/WFST_square.tess')
    #skymaps,plans,gw_triggers,fields = get_basics(event_list)
    start,end = 30,60
    Z = dict(svd_path='/home/Aujust/data/Kilonova/GPR/NN/',model_name = 'Bulla_3comp_spectra',ex_path = '/home/Aujust/data/Kilonova/GPR/Bulla_bns_spec_extp')
    Training_set= KNC.Training_set.Training_set(skymaps=skymaps[start:end],
                                                plans=plans[start:end],
                                                gw_triggers=gw_triggers[start:end],
                                                fields=fields,
                                                transient_names=['KN_POSSIS'], #'KN_POSSIS','KN_MOSFiT','SNIa','SNIbc','SNIIn','SNIIP','SNIa_91bg','SLSN'
                                                **Z)
    print('Import Done')
    with open('/home/Aujust/data/Kilonova/KN_classify/data/POSSIS_SED_angular.pkl','rb') as handle:
        POSSIS_SED_POOL = pickle.load(handle)
        handle.close()
        
    POSSIS_PARAM = POSSIS_SED_POOL['param']
    POSSIS_SED = POSSIS_SED_POOL['SED']
        
    history_name = 'Training_set_0908'
    save_name = 'Training_set_0908'
    history_dir = '/home/Aujust/data/Kilonova/KN_classify/data/' + history_name
    save_dir = '/home/Aujust/data/Kilonova/KN_classify/data/' + save_name
    kwargs = dict(basic=dict(n_workers=10,auto_save=True,save_dir='/home/Aujust/data/Kilonova/KN_classify/data/mock_inference_1201',
                             use_history=False),
                  KN_POSSIS=dict(SEDs=POSSIS_SED,n=100,n_models=15,store_params=True,params=POSSIS_PARAM),
                  KN_MOSFiT=dict(root='/home/Aujust/data/Kilonova/lc_samples/BNS/mosfit/mosfit_seds/',n=120,n_models=15),
                  SNIa=dict(n=15000),
                  SNIbc=dict(n=12000,zmax=0.99),
                  SNIIn=dict(n=12000,zmax=0.99),
                  SNIIP=dict(n=12000,zmax=0.99),
                  SNIa_91bg=dict(n=3000,n_models=10,root='/home/Aujust/data/Kilonova/KN_classify/data/SIMSED.SNIa-91bg/'),
                  SLSN=dict(n=300,n_models=5,root='/home/Aujust/data/Kilonova/KN_classify/data/SIMSED.SLSN-I-MOSFIT/'))
    dataset = Training_set.shuffle_skymaps(**kwargs)
    print('All set')
    
    # KN = {}
    # Other = {}

    # for obj_id in list(dataset.keys()):
    #     if dataset[obj_id].meta['class_num'] == 1:
    #         KN[obj_id] = dataset[obj_id]
    #     else:
    #         Other[obj_id] = dataset[obj_id]
            
    # print('Shuffle Done.')
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # for fname in ['KN','Other']:
    #     file = eval(fname)  
    #     if use_history:
    #         with open('{}/{}.pkl'.format(history_dir,fname),'rb') as handle:
    #             history_data_ = pickle.load(handle)
    #             handle.close()          
    #         file.update(history_data_)
            
    #     with open('{}/{}.pkl'.format(save_dir,fname),'wb') as handle:
    #         pickle.dump(file,handle)
    #         handle.close()
    # print()
