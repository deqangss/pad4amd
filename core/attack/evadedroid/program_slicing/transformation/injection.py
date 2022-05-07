# -*- coding: utf-8 -*-
"""
Applying a transformation into a malware app by injecting gadget into the APK
~~~~~~~~~~
This module that is originally created in [1] has been slightly modified for using in EvadeDroid.

[1] Intriguing Properties of Adversarial ML Attacks in the Problem Space 
    [S&P 2020], Pierazzi et al.
"""
import time
import logging
import pickle
from core.attack.evadedroid.settings import config as evadedroid_config
import os
from core.attack.evadedroid import utils as utils
from core.attack.evadedroid.utils import yellow, green, blue
from pprint import pformat


def transplant_organs(host, apks):
    # Create dictionary to store ongoing statistics
    results = {}
    # Load organs
    to_inject = {}
    for filename in apks:
        with open(filename + '/organ.p', 'rb') as f:
            o = pickle.load(f)
            # redirection: add by dli
            old_dir, old_path = o.location.split('/data/stored-components/')
            o.location = os.path.join(evadedroid_config['stored_components'], old_path)
            old_dir, old_path = o.donor_path.split('/data/apks/accessible/')
            o.donor_path = os.path.join(evadedroid_config['apks_accessible'], old_path)
        to_inject[o.feature] = o  

    # Calculate surplus permissions
    surplus_permissions = set()
    for organ in to_inject.values():
        surplus_permissions.update(organ.permissions)
    surplus_permissions -= set(host.permissions) 
    
    
    # Necessary features are known and extracted, perform inverse mapping
    logging.debug(green('Synthesizing adversarial evader...'))
    logging.info(green('Adding the following features:'))
    logging.info(green('\n' + '\n'.join(to_inject.keys())))
    logging.info(yellow('Including the following side-effects:'))
    side_effects = set()
    for organ in to_inject.values():
        organ_effects = {x for x in organ.feature_dict.keys()
                         if x != organ.feature}
        side_effects.update(organ_effects)
    logging.info(yellow('\n' + pformat(side_effects)))
    

    # These permissions are the ones needed for the new organs
    # They'll get added to the host manifest by the injector
    perm_file = os.path.join(host.tmpdname, 'permissions.txt')

    logging.info(
        'Injection requires ' + yellow(len(surplus_permissions)) + ' surplus permission(s): ' +
        yellow(surplus_permissions))
    logging.info(f'Writing to perm_file: {perm_file}...')

    
    if os.path.exists(perm_file):
        os.remove(perm_file)
    with open(perm_file, "wt") as f:
        for p in surplus_permissions:
            splits = p.split("::")[1].replace("_", ".").split(".")
            if len(splits) == 3:
                tmp_p = p.split("::")[1].replace("_", ".")
            elif len(splits) == 4:
                tmp_p = splits[0] + "." + splits[1] + "." + \
                        splits[2] + "_" + splits[3]
            elif len(splits) == 5:
                tmp_p = splits[0] + "." + splits[1] + "." + \
                        splits[2] + "_" + splits[3] + "_" + \
                        splits[4]
            else:
                tmp_p = ''
            f.write(tmp_p)
            
    # Create the string for input to the injector pointing to the single gadget folders
    apks = ','.join([o.location for o in to_inject.values()])
    logging.debug(f'Final organs to inplant: {apks}')
    
    # Move files into a working directory and perform injection
    now = time.time()
    # perm_file = perm_file if len(surplus_permissions) > 0 else None
    post_op_host, final_avg_cc, classes_final = transplant(host, apks, perm_file)
    post = time.time()

    results['time_injection'] = int(post - now)

    # Handle error results
    if 'error' in post_op_host:
        msg = f"Error occurred during injection {post_op_host}"
        #shutil.rmtree(host.tmpdname)
        print("Exception:" , msg)
        utils.perform_logging_for_attack("Exception:" + str(msg))
        return 1,msg,side_effects

    elif 'EXCEPTION' in post_op_host:
        logging.debug(" : " + post_op_host)
        logging.debug("Something went wrong during injection, see error.\n")
        utils.perform_logging_for_attack("Something went wrong during injection, see error.")
        if 'SootUtility.initSoot' in post_op_host:
            logging.debug("Soot exception for reading app")
            utils.perform_logging_for_attack("Soot exception for reading app")

        #shutil.rmtree(host.tmpdname)
        msg = "Something went wrong during injection, see error above."
        print("Exception:" , msg)
        utils.perform_logging_for_attack("Exception:" + str(msg))
        return 1, msg,side_effects

    # Resign the modified APK (will overwrite the unsigned one)
    #resign(post_op_host)
    logging.debug("Final apk signed")
    return 0,post_op_host,side_effects
    
def transplant(host, apks, perm_file=None):
    """Transplant a set of organs into a host malware.

    Args:
        host (Host): The host malware set to receive the transplanted organs.
        apks (str): Comma-separated list of donor APKs in the ice-box from which to transplant from.
        perm_file (str): The path to the permissions file of the host.

    Returns:
        (str, int, int): The path to the post-op host, its avg cc, its number of classes.
    """
    output_location = os.path.join(host.tmpdname, 'postop')

    host_path = os.path.join(host.tmpdname, host.name)
    if os.path.exists(os.path.join(output_location, host.name)):
        os.remove(os.path.join(output_location, host.name))
    os.makedirs(output_location, exist_ok=True)

    args = [host_path, apks,
            output_location,
            evadedroid_config['android_sdk'],
            evadedroid_config['mined_slices'],
            evadedroid_config['opaque_pred']]

    if perm_file:
        args.append(perm_file)
    logging.info(blue('Performing organ transplantation!'))
    out = utils.run_java_component(evadedroid_config['injector'], args)

    if out.find("Injection done") == -1:
        '''
        msg = "An error occurred during injection {} + {}: {}".format(
            host_path, apks, str(out))
        raise Exception(msg)
        '''
        print("injection failed: " + str(out))
        utils.perform_logging_for_attack("injection failed: " + str(out))
        return 'error',0,0


    avg_cc = 0
    classes = 0
    return os.path.join(output_location, host.name), int(avg_cc), int(classes)

def resign(app_path):
    """Resign the apk."""
    utils.run_java_component(evadedroid_config['resigner'], ['--overwrite', '-a', app_path])