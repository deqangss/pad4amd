# -*- coding: utf-8 -*-

"""
Classes for the core components needed for transplantation:
    * Host: The malware (transplant recipient) to be made adversarial.
    * Organ: An organ representing a code gadget made available for transplant.
~~~~~~~~~~~~~
This module that is originally created in [1] has been slightly modified for using in EvadeDroid.

[1] Intriguing Properties of Adversarial ML Attacks in the Problem Space 
    [S&P 2020], Pierazzi et al.

"""
import glob
import logging
import pickle
from pprint import pformat
import os
import shutil
import tempfile
from tqdm import tqdm
import core.attack.evadedroid.utils as utils
import core.attack.evadedroid.drebin.drebin as drebin
from core.attack.evadedroid.settings import config as evadedroid_config
from core.attack.evadedroid.utils import blue, green, cyan, magenta, yellow

hosts_dir = os.path.join(evadedroid_config['results_dir'], 'hosts')
os.makedirs(hosts_dir, exist_ok=True)
hosts_dir_temp = os.path.join(evadedroid_config['results_dir'], 'hosts_temp')
os.makedirs(hosts_dir_temp, exist_ok=True)


class Host:
    """The malware (transplant recipient) to be made adversarial."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        self.location = path

        # Extended part
        os.makedirs(evadedroid_config['tmp_dir'], exist_ok=True)
        self.results_dir = evadedroid_config['results_dir']
        os.makedirs(evadedroid_config['results_dir'], exist_ok=True)
        self.tmpdname = tempfile.mkdtemp(dir=evadedroid_config['tmp_dir'])
        self.size = os.path.getsize(path)
        self.features = drebin.get_features(path)
        self.permissions = [f for f in self.features if
                            'app_permissions' and 'android_permission' in f]
        self.dangerous_permissions = any([x in utils.dangerous_permissions
                                          for x in self.permissions])

    def cleanup(self):
        shutil.rmtree(self.tmpdname)

    def __str__(self):
        return (magenta('---- HOST MALWARE ----\n') +
                f'{cyan(self.name)} @ \n' +
                f'{blue(self.location)}  -> ' +
                f'{green(self.tmpdname)}\n' +
                f'size: {magenta(self.size)}B\n' +
                f'avg. cyclomatic complexity: {magenta(self.avg_cc)}\n' +
                f'classes: {blue(pformat(self.classes))}\n' +
                f'permissions: {magenta(pformat(self.permissions))}\n' +
                f'features: {cyan(pformat(self.features))}\n' +
                magenta('----------------------\n'))

    @classmethod
    def load(cls, path):
        hosts_file = os.path.join(hosts_dir, os.path.basename(path)) + '.p'
        if os.path.exists(hosts_file):
            with open(hosts_file, 'rb') as f:
                return pickle.load(f)
        host = Host(path)
        with open(hosts_file, 'wb') as f:
            pickle.dump(host, f)
            return host

    @classmethod
    def load_temp(cls, path):
        hosts_file = os.path.join(hosts_dir_temp, os.path.basename(path)) + '.p'
        if os.path.exists(hosts_file):
            with open(hosts_file, 'rb') as f:
                return pickle.load(f)
        host = Host(path)
        with open(hosts_file, 'wb') as f:
            pickle.dump(host, f)
            return host


class Organ:
    """An organ representing a code gadget made available for transplant."""

    def __init__(self, feature, donor_path):
        self.feature = feature
        self.donor_path = donor_path
        self.donor = utils.get_app_name(donor_path)
        self.location = locate_organ(feature, self.donor)
        self.feature_dict = {}
        self.permissions = set()
        self.classes = set()
        self.extraction_time = 0
        self.needs_vein = False
        self.dangerous_permissions = False

        # Extended part
        self.number_of_side_effect_features = 0

    def __str__(self):
        return (magenta('---- EXTRACTED ORGAN ----\n') +
                f'{cyan(self.feature)} @ ' +
                f'{blue(self.location)} \n' +
                f'donor: {self.donor} @ {self.donor_path}\n' +
                f'classes: {blue(pformat(self.classes))}\n' +
                f'permissions: {magenta(pformat(self.permissions))}\n' +
                f'feature dict: {cyan(pformat(self.feature_dict))}\n' +
                magenta('----------------------\n'))


def fetch_harvested(features):
    """Fetch all previously harvested organs for the given list of features."""
    harvested = []
    for feature in tqdm(features):
        organs = fetch_organs(feature)
        if organs:
            harvested.append(sorted(organs, key=lambda x: len(x.classes))[0])

    logging.info(yellow(f'{len(harvested)}/{len(features)} organs harvested and ready for transplant!'))
    return harvested


def fetch_organs(feature):
    """Find all previously harvested organs for a given feature.

    Note: We encountered some 'tricky globbers', features with a surplus of wildcards in their
    name which screw with the globbing so here we blacklist a few and otherwise try to catch as
    many as possible as a quick'n'dirty workaround.

    feature (str): Feature to fetch organs for.

    Returns:
        list: Organs available for the given feature.

    """

    # Special case to skip difficult features D:
    if feature == 'urls::http://([\\\\w-]+\\\\_)+[\\\\w-]+(/[\\\\w-\\\\_/?%=]*)?' or feature.startswith(
            'urls::http://******'):
        return []
    root = locate_organ(feature)
    path = f'{root}/**/*/organ.p'
    try:
        organ_locations = glob.glob(path, recursive=True)
    except:
        print('Tricky globber! ' + feature)
        return []
    organs = []
    for location in organ_locations:
        if not os.path.exists(location):
            pass
        else:
            with open(location, 'rb') as f:
                organs.append(pickle.load(f))
    return organs


def locate_organ(feature, app=None):
    """Locate singular organ for a given feature (and donor app, optionally)."""
    feature = feature.split('::')[-1] if '::' in feature else feature
    feature = utils.sanitize_url_feature_name(feature)

    feature = utils.sanitize_activity_feature_name(feature)

    path = os.path.join(evadedroid_config['ice_box'], feature)
    if app:
        # Make sure app is just the hash not a full path with ext
        # app = os.path.splitext(os.path.basename(app))[0]
        app = os.path.basename(app)
        path = os.path.join(path, app)
    return path
