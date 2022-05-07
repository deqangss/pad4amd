# -*- coding: utf-8 -*-

"""
A set of helper functions for interacting with the Drebin [1] feature extractor.
~~~~~~~~~

This module that is originally created in [2] has been slightly modified for using in EvadeDroid.


[1] DREBIN: Effective and Explainable Detection of Android Malware
    in Your Pocket [NDSS 2014], Arp et al.
    
[2] Intriguing Properties of Adversarial ML Attacks in the Problem Space 
    [S&P 2020], Pierazzi et al.

"""
import glob
import logging
import subprocess
import os
import shutil
import tempfile
import ujson as json
from core.attack.evadedroid.settings import config as evadedroid_config
from core.attack.evadedroid.settings import _absolute_project_path
from core.attack.evadedroid.utils import blue


def get_features(app_path):
    """Extract Drebin feature vectors from the app.

    Args:
        app_path: The app to extract features from.

    Returns:
        dict: The extracted feature vector in dictionary form.

    """
    app_path = os.path.abspath(app_path)
    output_dir = tempfile.mkdtemp(dir=evadedroid_config['tmp_dir'])
    cmd = ['python', evadedroid_config['project_root']+'/drebin_feature_extractor/drebin.py', app_path, output_dir]
    location = evadedroid_config['feature_extractor']
    logging.info(blue('Running command') + f' @ \'{location}\': {" ".join(cmd)}')
    subprocess.call(cmd, cwd=location)
    results_file = glob.glob(output_dir + '/results/*.json')[0]
    logging.debug('Extractor results in: {}'.format(results_file))
    with open(results_file, 'rt') as f:
        results = json.load(f)
    shutil.rmtree(output_dir)    
    return results


def to_j_feature(full_d_feature):
    feature_type, d_feature = full_d_feature.split('::')    
    if feature_type.find("app_permissions") != -1:
        return feature_type, d_feature.replace('android_permission_', 'android.permission.').replace("b'","").replace("'","").replace(" ","")
    else:
        return feature_type, d_feature.replace('_', '.').replace("b'","").replace("'","").replace(" ","")

