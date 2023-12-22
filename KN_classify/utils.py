import sys
import os
import logging
from multiprocessing import Pool, freeze_support, RLock, current_process
#sys.path.append('/home/Aujust/data/Kilonova/Constraint/')
sys.path.append('/home/Aujust/data/Kilonova/GPR/')
import toolkit as tkk
import time
from Knust import Knust
import simsurvey
import tqdm
import re
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import healpy as hp
import sncosmo
import numpy as np
import pandas as pd
import pickle
import copy
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import astropy.constants as const
from .Survey_constraintor import *
from .Survey_constraintor import Survey_constraintor as SSR


try:
    wfst_bands = tkk.load_wfst_bands()
except:
    pass
class_name_map = {
    1:'KN',
    2:'Other'
}

'ZP = 26.'

def event_mjd(event_id):
    'event_id[str]: ID should be conicide with IDs in GraceDb and return the trigger time in MJD'
    url = 'https://gracedb.ligo.org/superevents/{}/view/#event-information'.format(event_id)
    response = requests.get(url)
    soup = BeautifulSoup(response.text,'html.parser')
    html_str = str(soup.find_all('time')[1])
    soupt = BeautifulSoup(html_str,'html.parser')
    return Time(soupt.time['utc'][:-4],format='iso',scale='utc').mjd


#--------------------------------------#
#               Constant               #
#--------------------------------------#

day2sec = u.day.cgs.scale
MPC_CGS = u.Mpc.cgs.scale
C_CGS = const.c.cgs.value
M_SUN_CGS = const.M_sun.cgs.value
G_CGS = const.G.cgs.value
Jy = u.Jy.cgs.scale
ANG_CGS = u.Angstrom.cgs.scale
pi = np.pi
pc10 = 10 * u.pc.cgs.scale
SB_CGS = const.sigma_sb.cgs.value
H_CGS = const.h.cgs.value
K_B_CGS = const.k_B.cgs.value
KM_CGS = u.km.cgs.scale