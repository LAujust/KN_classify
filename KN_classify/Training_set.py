from .utils import *


class Training_set(object):
    '''
    skymaps[list]: A list of GW skymaps to make dataset.
    plans[list]: A list of survey plan for correspponding GW events.
    transient_names[list]: A list of tr name in (KN_POSSIS, KN_MOSFiT, SNIa, SNIbc, SNIa_91bg, SNIIn, SNIIP)
    fields[file]: Fields of WFST telescope.
    out_dir[str]: Output dir for dataset.
    '''
    def __init__(self,skymaps,plans,transient_names,fields,gw_triggers=None,out_dir='.',**kwgs):
        self.skymaps = skymaps
        self.skymap_nums = len(skymaps)
        self.plans = plans
        self.transient_names = transient_names
        self.fields = fields
        self.out_dir = out_dir
        self.kwgs = kwgs
        self.traning_set = {}
        self.gw_triggers = gw_triggers
        self.POSSIS_spec_model = None
        self.dust = sncosmo.CCM89Dust()
        self.basic_gw_info = self._gw_dict_list()
        
        try:
            self.svd_path = kwgs['svd_path']
            self.model_name = kwgs['model_name']
            self.ex_path = kwgs['ex_path']
        except:
            raise NameError('Input valid path dir for POSSIS surrogator')
        
        #=====LOGGING=====#
        root_dir = os.getcwd()
        log_fname = root_dir + '/' + time.strftime("%Y_%m_%d_%H_%M", time.localtime()) + '.log'
        logging.basicConfig(filename=log_fname, level=logging.INFO)
        logging.info('This log file was created at {}.'.format(time.strftime("%Y-%m-%d %H:%M", time.localtime())))
        
        
       
    #===== Methods =====# 
    def shuffle_skymaps(self,**kwgs):
        '''
        kwgs[dict]: A dict of dict of basic kwargs for each transient_name (e.g. except for skymap and plan)
        It should have a sctruture:
        {
            'basic':{
                'n_workers':num,
                'auto_save':False  If True, you should provide the path for saving files.
                'save_path':None
                ...
            },
            {
                'transeint_name':{
                    'kwgs for method transient_name':...
                }                
            },
        }
        '''
        TOTAL = {1:0,2:0}
        t0 = time.time()
        n_workers = kwgs['basic']['n_workers']
        auto_save = kwgs['basic'].get('auto_save',False)
        save_dir = kwgs['basic'].get('save_dir',None)
        use_history = kwgs['basic'].get('use_history',False)
        history_dir = kwgs['basic'].get('save_dir',save_dir)
        result = {}
        print('==============={} Skymaps, {} workers================='.format(self.skymap_nums,n_workers))
        for transient_name in self.transient_names:
            print('Processing {}.'.format(transient_name))
            
            kwgs_tr = kwgs[transient_name]
            kwgs_list = [0] * self.skymap_nums
            for i in range(self.skymap_nums):
                sp_dict = copy.deepcopy(kwgs_tr)
                sp_dict.update(self.basic_gw_info[i])
                kwgs_list[i] = sp_dict
                
            func = eval('self.{}'.format(transient_name))
            with Pool(processes=n_workers,initializer=tqdm.set_lock,initargs=(tqdm.get_lock(),)) as p:
                out = p.map(func,kwgs_list)
                p.close()
                p.join()
            print('\n'*5)
            merge_out = {}
            for dc in out:
                merge_out.update(dc)
            t_now = time.time()
            duration = t_now-t0
            t0 = t_now
            message = 'Processing Done, {} {} added. Running time: {:.2f}s.\n================================'.format(len(list(merge_out.keys())),transient_name,duration)
            print(message)
            logging.info(message)
            if auto_save and save_dir is not None:
                print('Auto saving...')
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                if use_history:
                    with open('{}/{}_raw.pkl'.format(history_dir,transient_name),'rb') as handle:
                        history_data_ = pickle.load(handle)
                        handle.close()          
                    merge_out.update(history_data_)
                message = 'Total {} {}.\n'.format(len(list(merge_out.keys())),transient_name,duration)
                logging.info(message)
                with open('{}/{}_raw.pkl'.format(save_dir,transient_name),'wb') as handle:
                    pickle.dump(merge_out,handle)
                    handle.close()
            result.update(merge_out)
            
        class_nums = np.array([result[key].meta['class_num'] for key in list(result.keys())])
        unique, counts = np.unique(class_nums,return_counts=True)
        counts = dict(zip(unique, counts))
        TOTAL_SHOW = ['{}:{}'.format(class_name_map[key],counts[key]) for key in list(counts.keys())]
        TOTAL_SHOW = ', '.join(TOTAL_SHOW)
        logging.info('Shuffle Done, {}'.format(TOTAL_SHOW))
        logging.info('Saving to {}'.format(save_dir))
        return result
            
            
                
    def _gw_dict_list(self):
        return [dict(skymap=self.skymaps[i],plan=self.plans[i],gw_trigger=self.gw_triggers[i]) for i in range(self.skymap_nums)]
        
    
    def KN_POSSIS(self,kwgs):
        skymap = kwgs['skymap']
        plan = kwgs['plan']
        gw_trigger = kwgs['gw_trigger']
        n_models = kwgs.get('n_models',20)
        n = kwgs.get('n',1000)
        SEDs = kwgs['SEDs']
        mjd_range = (gw_trigger-0.1,gw_trigger+0.1)
        event_id = skymap['event_id']
        store_params = kwgs.get('store_params',False)
        
        if store_params:
            params = kwgs['params']
        
        out = {}
        idxs = np.random.choice(len(SEDs),n_models)
        sample_size = 0
        current = current_process()
        pos = current._identity[0]-1
        with tqdm(total=n_models,position=pos) as pbar:
            for idx in idxs:
                if store_params:
                    param = params[idx]
                else:
                    param = None
                    
                #phase, wave, cos_theta, flux = SEDs[idx]
                #source = AngularTimeSeriesSource(phase, wave, cos_theta, flux)
                phase, wave, flux = SEDs[idx]
                source = TimeSeriesSource(phase, wave, flux)
                
                model = sncosmo.Model(source=source,effects=[self.dust, self.dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
                transientprop = dict(lcmodel=model, lcsimul_func=random_parameters) #random_parameters_ang
                lcs = self.get_lcs_kn(transientprop,mjd_range,skymap,plan,n=n)
                try:
                    n_detected = len(lcs.lcs)

                    sample_size += n_detected
                    for obj_idx in range(n_detected):
                        data_df = lcs[obj_idx]
                        t = data_df['time'] - gw_trigger   # subtract or not ??? #
                        flux = data_df['flux']
                        fluxerr = data_df['fluxerr']
                        bands = data_df['band']
                        zp = data_df['zp']
                        mwebv = lcs.meta['mwebv_sfd98'][obj_idx]
                        ra = lcs.meta['ra'][obj_idx]
                        dec = lcs.meta['dec'][obj_idx]
                        z = lcs.meta['z'][obj_idx]
                        t0 = lcs.meta['t0'][obj_idx] - gw_trigger 
                        simsurvey_id = lcs.meta['idx_orig'][obj_idx]
                        flux_2_err = flux/fluxerr
                        photflag = [4096 if i>5 else 0 for i in flux_2_err]

                        'Get line-of-sight probability'
                        log_prob,distmu_i,distsigma_i = line_of_sight_log_prob(ra,dec,skymap)
                        if np.isinf(distmu_i):
                            distmu_i = 0
                        if np.isinf(log_prob):
                            log_prob = -99
                            
                        

                        info_i = Table((t,flux,fluxerr,bands,photflag),names=('time','flux','fluxErr','passband','photflag'),
                                    meta={'class_num':1,'distmu':distmu_i,'distsigma':distsigma_i,'log_prob':log_prob,'redshift':z,'t0':t0,'b':30,'param':param})
                        
                        mjd_first_detection = info_i[info_i['photflag']==4096]['time'].min()
                        info_i['photflag'][info_i['time']==mjd_first_detection] = 6144
                        
                        #============================#
                        #Alternative: Replace distmu,distsigma by conditional distance probability of line-of-sight(LoS)?
                        #distmu_LoS, distsigma_LoS
                        #============================#
                        
                        'https://astrorapid.readthedocs.io/en/latest/usage.html#train-your-own-classifier-with-your-own-data'

                        out['POSSIS'+'_'+str(event_id)+'_'+str(int(idx))+'_'+str(int(simsurvey_id))] = info_i
                except:
                    print(skymap['event_id'],idx)
                    
                pbar.update(1)
                
        return out
    
    def KN_MOSFiT(self,kwgs):
        skymap = kwgs['skymap']
        plan = kwgs['plan']
        gw_trigger = kwgs['gw_trigger']
        n_models = kwgs.get('n_models',20)
        n = kwgs.get('n',1000)
        root = kwgs['root']
        flist = os.listdir(root)
        out = {}
        mjd_range = (gw_trigger-0.1,gw_trigger+0.1)
        event_id = skymap['event_id']

        idxs = np.random.choice(flist,n_models)
        wave = np.linspace(200,20000,100)
        sample_size = 0
        current = current_process()
        pos = current._identity[0]-1
        with tqdm(total=n_models,position=pos) as pbar:
            for i,idx in enumerate(idxs):
                data = np.loadtxt('{}{}'.format(root,idx))
                phase = data[:,0]
                flux = data[:,1:]/(4*pi*pc10**2)

                source = TimeSeriesSource(phase, wave, flux)
                model = sncosmo.Model(source=source,effects=[self.dust, self.dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
                transientprop = dict(lcmodel=model, lcsimul_func=random_parameters)
                lcs = self.get_lcs_kn(transientprop,mjd_range,skymap,plan,n=n)
                try:
                    n_detected = len(lcs.lcs)
                    sample_size += n_detected
                    for obj_idx in range(n_detected):
                        data_df = lcs[obj_idx]
                        t = data_df['time'] - gw_trigger   # subtract or not ??? #
                        flux = data_df['flux']
                        fluxerr = data_df['fluxerr']
                        bands = data_df['band']
                        zp = data_df['zp']
                        mwebv = lcs.meta['mwebv_sfd98'][obj_idx]
                        ra = lcs.meta['ra'][obj_idx]
                        dec = lcs.meta['dec'][obj_idx]
                        z = lcs.meta['z'][obj_idx]
                        t0 = lcs.meta['t0'][obj_idx] - gw_trigger
                        simsurvey_id = lcs.meta['idx_orig'][obj_idx]
                        flux_2_err = flux/fluxerr
                        photflag = [4096 if i>5 else 0 for i in flux_2_err]

                        'Get line-of-sight probability'
                        log_prob,distmu_i,distsigma_i = line_of_sight_log_prob(ra,dec,skymap)
                        if np.isinf(distmu_i):
                            distmu_i = 0
                        if np.isinf(log_prob):
                            log_prob = -99

                        info_i = Table((t,flux,fluxerr,bands,photflag),names=('time','flux','fluxErr','passband','photflag'),
                                    meta={'class_num':1,'distmu':distmu_i,'distsigma':distsigma_i,'log_prob':log_prob,'redshift':z,'t0':t0,'b':30})
                        
                        mjd_first_detection = info_i[info_i['photflag']==4096]['time'].min()
                        info_i['photflag'][info_i['time']==mjd_first_detection] = 6144
                        
                        'https://astrorapid.readthedocs.io/en/latest/usage.html#train-your-own-classifier-with-your-own-data'
                        # info_i = {
                        #     'time':t,
                        #     'flux':flux,
                        #     'fluxerr':fluxerr,
                        #     'ra':ra,
                        #     'dec':dec,
                        #     'mwebv':mwebv,
                        #     'photflag':photflag,
                        #     'zp':zp,
                        #     'dL':dL,
                        #     'ddL':ddL
                        # }
                        out['mosfit'+'_'+str(event_id)+'_'+str(int(i))+'_'+str(int(simsurvey_id))] = info_i
                except:
                    print(skymap['event_id'],idx)
                pbar.update(1)
                
        return out

    
    def SNIa(self,kwgs):
        skymap = kwgs['skymap']
        plan = kwgs['plan']
        gw_trigger = kwgs['gw_trigger']
        n_models = kwgs.get('n_models',20)
        n = kwgs.get('n',1000)
        transient = kwgs.get('transient','Ia')
        template = kwgs.get('template','hsiao')
        zmax = kwgs.get('zmax',0.9)
        event_id = skymap['event_id']
                
        out = {}
        RA = plan.pointings['RA']
        RA[RA<0] += 360
        ra_range = (RA.min(),RA.max())
        dec_range=(plan.pointings['Dec'].min(),plan.pointings['Dec'].max())
        mjd_range = (gw_trigger-50,gw_trigger+20)
        lcs = self.get_lcs_sn(transient=transient,template=template,mjd_range=mjd_range,ra_range=ra_range,dec_range=dec_range,plan=plan,n=n,zmax=zmax)
        n_detected = len(lcs.lcs)

        current = current_process()
        pos = current._identity[0]-1
        init_num = np.random.randint(10000)
        with tqdm(total=n_detected,position=pos) as pbar:
            for obj_idx in range(n_detected):
                data_df = lcs[obj_idx]
                t = data_df['time'] - gw_trigger  # subtract or not ??? #
                flux = data_df['flux']
                fluxerr = data_df['fluxerr']
                bands = data_df['band']
                zp = data_df['zp']
                mwebv = lcs.meta['mwebv_sfd98'][obj_idx]
                ra = lcs.meta['ra'][obj_idx]
                dec = lcs.meta['dec'][obj_idx]
                simsurvey_id = lcs.meta['idx_orig'][obj_idx]
                z = lcs.meta['z'][obj_idx]
                t0 = lcs.meta['t0'][obj_idx] - gw_trigger
                flux_2_err = flux/fluxerr
                photflag = [4096 if i>5 else 0 for i in flux_2_err]
                #photflag = np.zeros((len(t)))

                'Get line-of-sight probability'
                log_prob,distmu_i,distsigma_i = line_of_sight_log_prob(ra,dec,skymap)
                if np.isinf(distmu_i):
                    distmu_i = 0
                if np.isinf(log_prob):
                    log_prob = -99
                        
                info_i = Table((t,flux,fluxerr,bands,photflag),names=('time','flux','fluxErr','passband','photflag'),
                                    meta={'class_num':2,'distmu':distmu_i,'distsigma':distsigma_i,'log_prob':log_prob,'mwebv':mwebv,'redshift':z,'t0':t0,'b':30})
                
                mjd_first_detection = info_i[info_i['photflag']==4096]['time'].min()
                info_i['photflag'][info_i['time']==mjd_first_detection] = 6144
                        
                #============================#
                #Alternative: Replace distmu,distsigma by conditional distance probability of line-of-sight(LoS)?
                #distmu_LoS, distsigma_LoS
                #============================#
                        
                'https://astrorapid.readthedocs.io/en/latest/usage.html#train-your-own-classifier-with-your-own-data'

                out['SN'+transient+'_'+str(event_id)+'_'+str(int(simsurvey_id))] = info_i
                        
                pbar.update(1)
                
        return out
    
    def SNIa_91bg(self,kwgs):
        skymap = kwgs['skymap']
        plan = kwgs['plan']
        gw_trigger = kwgs['gw_trigger']
        n_models = kwgs.get('n_models',20)
        n = kwgs.get('n',1000)
        root = kwgs['root']
        ratefunc = kwgs.get('ratefunc',lambda z: 5e-1 * (1+z)**(1.5))
        event_id = skymap['event_id']
    
        out = {}
        RA = plan.pointings['RA']
        RA[RA<0] += 360
        ra_range = (RA.min(),RA.max())
        dec_range=(plan.pointings['Dec'].min(),plan.pointings['Dec'].max())
        mjd_range = (gw_trigger-50,gw_trigger+20)
        flist = os.listdir(root)[1:]
        idxs = np.random.choice(flist,n_models)
        current = current_process()
        pos = current._identity[0]-1
        with tqdm(total=n_models,position=pos) as pbar:
            for i,idx in enumerate(idxs):
                dat = np.loadtxt('{}{}'.format(root,idx))
                T, WAVE, FLUX = dat[:,0],dat[:,1],dat[:,2]
                phase = np.unique(T)
                wave = np.unique(WAVE)
                flux = FLUX.reshape((len(phase),len(wave)))
                source = TimeSeriesSource(phase, wave, flux)
                model = sncosmo.Model(source=source,effects=[self.dust, self.dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
                transientprop = dict(lcmodel=model, lcsimul_func=random_parameters)
                lcs = self.get_lcs_sn_sed(transientprop=transientprop,plan=plan,mjd_range=mjd_range,ra_range=ra_range,dec_range=dec_range,n=n,ratefunc=ratefunc)
                n_detected = len(lcs.lcs)
                for obj_idx in range(n_detected):
                    data_df = lcs[obj_idx]
                    t = data_df['time'] - gw_trigger   # subtract or not ??? #
                    flux = data_df['flux']
                    fluxerr = data_df['fluxerr']
                    bands = data_df['band']
                    zp = data_df['zp']
                    mwebv = lcs.meta['mwebv_sfd98'][obj_idx]
                    ra = lcs.meta['ra'][obj_idx]
                    dec = lcs.meta['dec'][obj_idx]
                    simsurvey_id = lcs.meta['idx_orig'][obj_idx]
                    z = lcs.meta['z'][obj_idx]
                    t0 = lcs.meta['t0'][obj_idx] - gw_trigger
                    flux_2_err = flux/fluxerr
                    photflag = [4096 if i>5 else 0 for i in flux_2_err]
                    #photflag = np.zeros((len(t)))

                    'Get line-of-sight probability'
                    log_prob,distmu_i,distsigma_i = line_of_sight_log_prob(ra,dec,skymap)
                    if np.isinf(distmu_i):
                        distmu_i = 0
                    if np.isinf(log_prob):
                        log_prob = -99
                                
                    info_i = Table((t,flux,fluxerr,bands,photflag),names=('time','flux','fluxErr','passband','photflag'),
                                            meta={'class_num':2,'distmu':distmu_i,'distsigma':distsigma_i,'log_prob':log_prob,'mwebv':mwebv,'redshift':z,'t0':t0,'b':30})
                        
                    mjd_first_detection = info_i[info_i['photflag']==4096]['time'].min()
                    info_i['photflag'][info_i['time']==mjd_first_detection] = 6144
                                
                        #============================#
                        #Alternative: Replace distmu,distsigma by conditional distance probability of line-of-sight(LoS)?
                        #distmu_LoS, distsigma_LoS
                        #============================#
                                
                    'https://astrorapid.readthedocs.io/en/latest/usage.html#train-your-own-classifier-with-your-own-data'

                    out['SNIa-91bg'+'_'+str(event_id)+'_'+str(int(i))+'_'+str(int(simsurvey_id))] = info_i
                        
                pbar.update(1)
                
        return out
    
    def SLSN(self,kwgs):
        skymap = kwgs['skymap']
        plan = kwgs['plan']
        gw_trigger = kwgs['gw_trigger']
        n_models = kwgs.get('n_models',20)
        n = kwgs.get('n',1000)
        root = kwgs['root']
        ratefunc = kwgs.get('ratefunc',lambda z: 5e-1 * (1+z))
        event_id = skymap['event_id']      
        
        out = {}
        RA = plan.pointings['RA']
        RA[RA<0] += 360
        ra_range = (RA.min(),RA.max())
        dec_range=(plan.pointings['Dec'].min(),plan.pointings['Dec'].max())
        mjd_range = (gw_trigger-50,gw_trigger+20)
        flist = os.listdir(root)
        flist = [item for item in flist if item[:4] == 'slsn']
        idxs = np.random.choice(flist,n_models)
        current = current_process()
        pos = current._identity[0]-1
        with tqdm(total=n_models,position=pos) as pbar:
            for i,idx in enumerate(idxs):
                dat = np.loadtxt('{}{}'.format(root,idx))
                T, WAVE, FLUX = dat[:,0],dat[:,1],dat[:,2]/(4*pi*pc10**2)
                phase = np.unique(T)
                wave = np.unique(WAVE)
                flux = FLUX.reshape((len(phase),len(wave)))
                source = TimeSeriesSource(phase, wave, flux)
                model = sncosmo.Model(source=source,effects=[self.dust, self.dust], effect_names=['host', 'MW'], effect_frames=['rest', 'obs'])
                transientprop = dict(lcmodel=model, lcsimul_func=random_parameters)
                lcs = self.get_lcs_sn_sed(transientprop=transientprop,plan=plan,mjd_range=mjd_range,ra_range=ra_range,dec_range=dec_range,n=n,ratefunc=ratefunc)
                n_detected = len(lcs.lcs)
                for obj_idx in range(n_detected):
                    data_df = lcs[obj_idx]
                    t = data_df['time'] - gw_trigger   # subtract or not ??? #
                    flux = data_df['flux']
                    fluxerr = data_df['fluxerr']
                    bands = data_df['band']
                    zp = data_df['zp']
                    mwebv = lcs.meta['mwebv_sfd98'][obj_idx]
                    ra = lcs.meta['ra'][obj_idx]
                    dec = lcs.meta['dec'][obj_idx]
                    simsurvey_id = lcs.meta['idx_orig'][obj_idx]
                    z = lcs.meta['z'][obj_idx]
                    t0 = lcs.meta['t0'][obj_idx] - gw_trigger
                    flux_2_err = flux/fluxerr
                    photflag = [4096 if i>5 else 0 for i in flux_2_err]
                    #photflag = np.zeros((len(t)))

                    'Get line-of-sight probability'
                    log_prob,distmu_i,distsigma_i = line_of_sight_log_prob(ra,dec,skymap)
                    if np.isinf(distmu_i):
                        distmu_i = 0
                    if np.isinf(log_prob):
                        log_prob = -99
                                
                    info_i = Table((t,flux,fluxerr,bands,photflag),names=('time','flux','fluxErr','passband','photflag'),
                                            meta={'class_num':2,'distmu':distmu_i,'distsigma':distsigma_i,'log_prob':log_prob,'mwebv':mwebv,'redshift':z,'t0':t0,'b':30})
                        
                    mjd_first_detection = info_i[info_i['photflag']==4096]['time'].min()
                    info_i['photflag'][info_i['time']==mjd_first_detection] = 6144
                                
                        #============================#
                        #Alternative: Replace distmu,distsigma by conditional distance probability of line-of-sight(LoS)?
                        #distmu_LoS, distsigma_LoS
                        #============================#
                                
                    'https://astrorapid.readthedocs.io/en/latest/usage.html#train-your-own-classifier-with-your-own-data'

                    out['SLSN'+'_'+str(event_id)+'_'+str(int(i))+'_'+str(int(simsurvey_id))] = info_i
                        
                pbar.update(1)  
        return out
    
    def SNIbc(self,kwgs):
        kwgs['transient'] = 'Ibc'
        kwgs['template'] = 'nugent'
        return self.SNIa(kwgs)
    
    def SNIIn(self,kwgs):
        kwgs['transient'] = 'IIn'
        kwgs['template'] = 'nugent'
        return self.SNIa(kwgs)
    
    def SNIIP(self,kwgs):
        kwgs['transient'] = 'IIP'
        kwgs['template'] = 'nugent'
        return self.SNIa(kwgs)
        
        
    def load_tf_model(self,svd_path,model_name,ex_path):
        self.POSSIS_spec_model = Knust(model_type='tensorflow',model_dir=svd_path,model_name=model_name)
        self.POSSIS_ex_model = Extp(ex_path)
        
    def get_lcs_kn(self,transientprop,mjd_range,skymap,plan,n=1000,zmax=0.2):
        tr = simsurvey.get_transient_generator([0,zmax],
                                                ntransient=n,
                                                ratefunc=lambda z: 5e-1 * (1+z),
                                                sfd98_dir=sfd98_dir,
                                                transientprop=transientprop,
                                                mjd_range=mjd_range,
                                                skymap=skymap
                                                )
        survey = simsurvey.SimulSurvey(generator=tr, plan=plan, n_det=2, threshold=5., sourcenoise=True)
        
        lcs = survey.get_lightcurves(progress_bar=True)
        return lcs
    
    def get_lcs_sn(self,transient,template,mjd_range,ra_range,dec_range,plan,n=1000,zmax=0.9):

        tr = simsurvey.get_transient_generator([0,zmax],
                                                transient=transient,
                                                template=template,
                                                ntransient=n,
                                                sfd98_dir=sfd98_dir,
                                                mjd_range=mjd_range,
                                                ra_range=ra_range,
                                                dec_range=dec_range
                                                )
        survey = simsurvey.SimulSurvey(generator=tr, plan=plan, n_det=2, threshold=5., sourcenoise=True)
        
        lcs = survey.get_lightcurves(progress_bar=True)
        return lcs
    
    def get_lcs_sn_sed(self,transientprop,mjd_range,ra_range,dec_range,plan,n=1000,zmax=0.9,ratefunc=lambda z: 5e-1 * (1+z)**(1.5)):
        
        tr = simsurvey.get_transient_generator([0,zmax],
                                                ntransient=n,
                                                ratefunc=ratefunc,
                                                sfd98_dir=sfd98_dir,
                                                transientprop=transientprop,
                                                mjd_range=mjd_range,
                                                ra_range=ra_range,
                                                dec_range=dec_range
                                                )
        survey = simsurvey.SimulSurvey(generator=tr, plan=plan, n_det=2, threshold=5., sourcenoise=True)
        
        lcs = survey.get_lightcurves(progress_bar=True)
        return lcs

    
    

def plot_lc(lc_data):
    colors = ['b','g','r']
    bands = np.atleast_1d(pd.unique(lc_data['passband']))
    len_ = len(bands)
        
    if len_ > 1:
        fig, ax = plt.subplots(1,len_,figsize=(len_*5,3))
        for i in range(len_):
            band = bands[i]
            data_plot = lc_data[lc_data['passband']==band]
            mjd = data_plot['time']
            flux = data_plot['flux']
            err = data_plot['fluxErr']
            ax[i].errorbar(mjd,flux,err,fmt='.',color=colors[i])
            ax[i].set_xlim([-0.,3])
            ax[i].set_xlabel('Phase')
            ax[i].set_ylabel('Flux')
            ax[i].plot(np.linspace(0,3,20),np.zeros((20)),linestyle='--')
            
    else:
        fig, ax = plt.subplots(figsize=(len_*5,3))
        i = 0
        band = bands[i]
        data_plot = lc_data[lc_data['passband']==band]
        mjd = data_plot['time']
        flux = data_plot['flux']
        err = data_plot['fluxErr']
        ax.errorbar(mjd,flux,err,fmt='.',color=colors[i])
        ax.plot(np.linspace(0,3,20),np.zeros((20)),linestyle='--')
        ax.set_xlim([-0.,3])
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')  

def line_of_sight_log_prob(ra,dec,skymap):
    prob, distmu, distsigma = skymap['prob'], skymap['distmu'], skymap['distsigma']
    npix = len(prob)
    nside = hp.npix2nside(npix)
    theta = 0.5 * np.pi - np.deg2rad(dec)
    phi = np.deg2rad(ra)
    ipix = hp.ang2pix(nside, theta, phi)
    pixarea = hp.nside2pixarea(nside)
    dp_dA = prob[ipix] / pixarea
    distmu_los,distsigma_los = distmu[ipix],distsigma[ipix]

    return np.log10(dp_dA),distmu_los,distsigma_los    
            
            


    