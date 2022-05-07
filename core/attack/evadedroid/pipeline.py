import sys
import timeit
import json
import numpy as np
from itertools import repeat
import pickle
import os
import io
import shutil
import torch

mp = torch.multiprocessing.get_context('forkserver')

from core.attack.evadedroid.program_slicing.transformation import extraction
from core.attack.evadedroid import utils
from core.attack.evadedroid import evasion
from core.attack.evadedroid.program_slicing import transformation
from core.attack.evadedroid.settings import config as evadedroid_config

sys.path.append(evadedroid_config['transformation'])


def perturb(model, number_of_query, malware_apps_path, is_hard_label, model_type, serial=True):
    utils.perform_logging(
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start: evasion attack on pad4amd_{model_type} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~ Start: evasion attack on pad4amd_{model_type}  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    action_set_path = os.path.join(evadedroid_config['stored_components'], 'action_set.p')
    with open(action_set_path, 'rb') as f:
        action_set = pickle.load(f)
    for item in action_set.keys():
        organ_path = action_set[item]
        # redirection: add by dli
        old_dir, old_path = organ_path.split('/data/stored-components/')
        organ_path = os.path.join(evadedroid_config['stored_components'], old_path)
        with open(organ_path, 'rb') as f:
            organ = pickle.load(f)
        # redirection: add by dli
        old_dir, old_path = organ.location.split('/data/stored-components/')
        organ.location = os.path.join(evadedroid_config['stored_components'], old_path)
        old_dir, old_path = organ.donor_path.split('/data/apks/accessible/')
        organ.donor_path = os.path.join(evadedroid_config['apks_accessible'], old_path)
        action_set[item] = organ
    # number_of_query = 20
    # base_size = 0.1
    base_size = 4.
    malware_detector = "pad4amd_{}".format(model_type)
    hard_label = is_hard_label
    path_base = os.path.join(evadedroid_config['results_dir'], 'EvadeDroid/{}'.format(malware_detector))
    if os.path.isdir(path_base) == False:
        os.makedirs(path_base, exist_ok=True)
    print("len(malware_apps_path): ", len(malware_apps_path))
    s = 5
    if hard_label == True:
        hardlabel = 1
    else:
        hardlabel = 0
    increase_in_size = base_size * s
    name = "result-noquery_%d-size_%f-hardlabel_%d" % (number_of_query, increase_in_size, hardlabel)
    path = os.path.join(path_base, name)
    if os.path.isdir(path) == False:
        os.mkdir(path)
    print("increase_in_size: ", increase_in_size)
    if serial == True:
        for app_path in malware_apps_path:
            do_black_box_attack_for_DNN(model, app_path, action_set, number_of_query, increase_in_size,
                                        hard_label, malware_detector, path)
    else:
        with mp.Pool(processes=evadedroid_config['nprocs_evasion']) as p:
            p.starmap(do_black_box_attack_for_DNN, zip(repeat(model),
                                                       malware_apps_path,
                                                       repeat(action_set),
                                                       repeat(number_of_query),
                                                       repeat(increase_in_size),
                                                       repeat(hard_label),
                                                       repeat(malware_detector),
                                                       repeat(path)))
    print("Finish attacking  ...")
    if s != 5:
        shutil.rmtree(os.path.join(evadedroid_config['results_dir'], 'hosts'))
        shutil.rmtree(os.path.join(evadedroid_config['results_dir'], 'postop'))
        os.mkdir(os.path.join(evadedroid_config['results_dir'], 'hosts'))
        os.mkdir(os.path.join(evadedroid_config['results_dir'], 'postop'))
    else:
        os.rename(os.path.join(evadedroid_config['results_dir'], 'hosts'), os.path.join(evadedroid_config['results_dir'],
                                                                             'hosts-pad4amd-{}'.format(model_type) + '-noquery_%d-size_%f-hardlabel_%d' % (
                                                                                 number_of_query, increase_in_size,
                                                                                 hardlabel)))
        os.rename(os.path.join(evadedroid_config['results_dir'], 'postop'), os.path.join(evadedroid_config['results_dir'],
                                                                              'postop-pad4amd-%s-noquery_%d-size_%f-hardlabel_%d' % (
                                                                                             model_type,
                                                                                  number_of_query, increase_in_size,
                                                                                  hardlabel)))
        os.mkdir(os.path.join(evadedroid_config['results_dir'], 'hosts'))
        os.mkdir(os.path.join(evadedroid_config['results_dir'], 'postop'))

    utils.perform_logging(
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~ End: evasion attack on pad4amd_{model_type} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~ End: evasion attack on pad4amd_{model_type} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def do_black_box_attack(app_path, action_set, number_of_query, increase_in_size,
                        model_inaccessible, hard_label, malware_detector, path):
    path_fail = os.path.join(evadedroid_config['stored_components'], 'malware_apk_fail.p')
    with open(path_fail, 'rb') as f:
        malware_apk_fail = pickle.load(f)
    if os.path.basename(app_path) in malware_apk_fail:
        print("app is corrupt")
        return

    # path_base = os.path.join(evadedroid_config['results_dir'],'EvadeDroid/Drebin','result-noquery_%d-size_%f-hardlabel_%d'%(number_of_query,increase_in_size,(hard_label * 1)))
    if malware_detector == "Drebin" or malware_detector == "SecSVM" or malware_detector == "MaMaDroid":
        path_base = os.path.join(evadedroid_config['results_dir'], 'EvadeDroid/%s' % (malware_detector),
                                 'result-noquery_%d-size_%f-hardlabel_%d' % (
                                 number_of_query, increase_in_size, (hard_label * 1)))
    else:
        if malware_detector == "ESET-NOD32":
            name = "result_%s" % ("ESETNOD32")
        else:
            name = "result_%s" % (malware_detector)
        path_base = os.path.join(evadedroid_config['results_dir'], 'EvadeDroid/VirusTotal/%s' % (name))
    # path_base_temp = path_base +'/back'
    apps_checked = os.listdir(path_base)

    if os.path.splitext(os.path.basename(app_path))[0] + '.p' in apps_checked:
        print('%s has been already checked' % (os.path.basename(app_path)))
        return

    '''
    app_temp = os.path.splitext(os.path.basename(app_path))[0] +'.p'
    apk_info_path = os.path.join(path_base,app_temp)
    with open(apk_info_path , 'rb') as f:
        apk = pickle.load(f)
    if apk.adv_malware_label == 0:
        print("app has been already manipulated successfully")
        return

   '''

    print("----------------------------------------------------")
    no_finished_apps = len(os.listdir(path))
    print("no_finished_apps = " + str(no_finished_apps))

    malware = app_path  # os.path.join(evadedroid_config['apks_accessible'],'malware',malware_app[i])

    apk = evasion.generate_adversarial_example(malware, action_set, number_of_query,
                                               increase_in_size, model_inaccessible,
                                               hard_label, malware_detector)
    print("app_name = ", apk.app_name)
    print("malware_label = ", apk.malware_label)
    print("adv_malware_label = ", apk.adv_malware_label)
    print("number_of_queries = ", apk.number_of_queries)
    print("percentage_increasing_size = ", apk.percentage_increasing_size)
    print("number_of_features_malware = ", apk.number_of_features_malware)
    print("number_of_features_adv_malware = ", apk.number_of_features_adv_malware)
    print("number_of_features_adv_malware_per_query = ", apk.number_of_features_adv_malware_per_query)
    print("number_of_api_calls_malware = ", apk.number_of_api_calls_malware)
    print("number_of_api_calls_adv_malware = ", apk.number_of_api_calls_adv_malware)
    print("number_of_api_calls_adv_malware_per_query = ", apk.number_of_api_calls_adv_malware_per_query)
    print("transformations = ", apk.transformations)
    print("intact_due_to_soot_error = ", apk.intact_due_to_soot_error)
    print("execution_time =  ", apk.execution_time)
    print("classified_with_hard_label = ", apk.classified_with_hard_label)

    # apk_info_path = os.path.join(path,apk.app_name.replace('.apk','.p'))
    apk_info_path = os.path.join(path, os.path.splitext(apk.app_name)[0] + '.p')
    with open(apk_info_path, 'wb') as f:
        pickle.dump(apk, f)
        print("copy done: %s" % (apk_info_path))
    print("----------------------------------------------------")


def do_black_box_attack_for_DNN(model, app_path, action_set, number_of_query, increase_in_size,
                                hard_label, malware_detector, path):
    print("app_path: ", app_path)
    path_fail = os.path.join(evadedroid_config['stored_components'], 'malware_apk_fail.p')
    with open(path_fail, 'rb') as f:
        malware_apk_fail = pickle.load(f)
    if os.path.basename(app_path) in malware_apk_fail:
        print("app is corrupt")
        return

    path_base = os.path.join(evadedroid_config['results_dir'],
                             'EvadeDroid/%s' % (malware_detector),
                             'result-noquery_%d-size_%f-hardlabel_%d' % (
                             number_of_query, increase_in_size, (hard_label * 1)))
    apps_checked = os.listdir(path_base)

    if os.path.splitext(os.path.basename(app_path))[0] + '.p' in apps_checked:
        print('%s has been already checked' % (os.path.basename(app_path)))
        return
    os.makedirs(path_base, exist_ok=True)
    print("----------------------------------------------------")
    no_finished_apps = len(os.listdir(path))
    print("no_finished_apps = " + str(no_finished_apps))

    malware = app_path  # os.path.join(evadedroid_config['apks_accessible'],'malware',malware_app[i])

    apk = evasion.generate_adversarial_example(malware, action_set, number_of_query,
                                               increase_in_size, model,
                                               hard_label, malware_detector)
    print("app_name = ", apk.app_name)
    print("malware_label = ", apk.malware_label)
    print("adv_malware_label = ", apk.adv_malware_label)
    print("number_of_queries = ", apk.number_of_queries)
    print("percentage_increasing_size = ", apk.percentage_increasing_size)
    print("number_of_features_malware = ", apk.number_of_features_malware)
    print("number_of_features_adv_malware = ", apk.number_of_features_adv_malware)
    print("number_of_features_adv_malware_per_query = ", apk.number_of_features_adv_malware_per_query)
    print("number_of_api_calls_malware = ", apk.number_of_api_calls_malware)
    print("number_of_api_calls_adv_malware = ", apk.number_of_api_calls_adv_malware)
    print("number_of_api_calls_adv_malware_per_query = ", apk.number_of_api_calls_adv_malware_per_query)
    print("transformations = ", apk.transformations)
    print("intact_due_to_soot_error = ", apk.intact_due_to_soot_error)
    print("execution_time =  ", apk.execution_time)
    print("classified_with_hard_label = ", apk.classified_with_hard_label)

    # apk_info_path = os.path.join(path,apk.app_name.replace('.apk','.p'))
    apk_info_path = os.path.join(path, os.path.splitext(apk.app_name)[0] + '.p')
    with open(apk_info_path, 'wb') as f:
        pickle.dump(apk, f)
        print("copy done: %s" % (apk_info_path))
    print("----------------------------------------------------")
