# -*- coding: utf-8 -*-

"""
Preparing action set   
"""
import logging
import pickle
import subprocess
import sys
import uuid
from datetime import datetime
from timeit import default_timer as timer
import os
import shutil
import tempfile
import core.attack.evadedroid.drebin.drebin as drebin
import core.attack.evadedroid.program_slicing.transformation.inpatients as inpatients
import core.attack.evadedroid.utils as utils
from core.attack.evadedroid.settings import config as evadedroid_config
from core.attack.evadedroid.utils import yellow, blue, green, red
from itertools import repeat
import torch

mp = torch.multiprocessing.get_context('forkserver')

action_set = dict()


def create_action_set(donors):
    serial = False
    apks = dict()
    apps = os.listdir(os.path.join(evadedroid_config['apks'], 'accessible/normal'))
    for app in apps:
        apks[app] = os.path.join(evadedroid_config['apks'], 'accessible/normal', app)

    print("len(donors): " + str(len(donors)))
    donor_id = 0
    for donor in donors:
        donor_id += 1
        donor_path = apks[donor]
        donor_dict = drebin.get_features(donor_path)
        features = donor_dict.keys()
        print("No api call features: ", len([f for f in features if 'api_calls' in f]))
        print("No api interesting features: ", len([f for f in features if 'interesting_calls' in f]))
        print("No api permissions: ", len([f for f in features if 'api_permissions' in f]))
        if serial == True:
            for feature in features:
                extract_an_action(feature, donor, donors, donor_path, donor_id)
        else:
            with mp.Pool(processes=evadedroid_config['nprocs_evasion']) as p:
                p.starmap(extract_an_action, zip(features,
                                                 repeat(donor),
                                                 repeat(donors),
                                                 repeat(donor_path),
                                                 repeat(donor_id)))

    organs = os.listdir(os.path.join(evadedroid_config['stored_components'], 'organs'))
    action_set = dict()
    for organ in organs:
        action_set[organ] = os.path.join(evadedroid_config['stored_components'], 'organs', organ)

    action_set_path = os.path.join(evadedroid_config['stored_components'], 'action_set.p')
    with open(action_set_path, 'wb') as f:
        pickle.dump(action_set, f)


def extract_an_action(feature, donor, donors, donor_path, donor_id):
    if feature.split("::")[0] == "interesting_calls" or feature.split("::")[0] == "api_calls" or feature.split("::")[
        0] == "api_permissions":
        utils.perform_logging(
            "feature: " + feature + " - donor " + str(donor_id) + "(" + donor + ")" + " out of " + str(
                len(donors)) + " dononrs")
        print("feature: " + feature + " - donor " + str(donor_id) + "(" + donor + ")" + " out of " + str(
            len(donors)) + " dononrs")

        print("Organ harvesting: start")
        organ = harvest_organ_from_donor(feature, donor_path, donor_id)
        print("Organ harvesting: end")
        if organ != None:
            utils.perform_logging(
                "organ harvesting corresponds " + feature + " from " + os.path.basename(donor_path) + " was successful")
            print(
                "organ harvesting corresponds " + feature + " from " + os.path.basename(donor_path) + " was successful")

            feature_name = feature.split('::')[-1] if '::' in feature else feature
            feature_name = utils.sanitize_url_feature_name(feature_name)
            feature_name = utils.sanitize_activity_feature_name(feature_name)
            organ_path = os.path.join(evadedroid_config['stored_components'], 'organs',
                                      "%(donor)s-%(feature)s_organ.p" % {"donor": donor, "feature": feature_name})
            with open(organ_path, 'wb') as f:
                pickle.dump(organ, f)

            # This function that is originally created in [1] has been modified for using in EvadeDroid


def harvest_organ_from_donor(feature, donor_path, donor_id):
    """Harvest feature from donor."""
    "interesting_calls::Read/Write External Storage"

    organ = inpatients.Organ(feature, donor_path)
    os.makedirs(organ.location, exist_ok=True)

    failure_test = os.path.join(organ.location, 'failed')
    pickle_location = os.path.join(organ.location, 'organ.p')

    if os.path.exists(failure_test):
        utils.perform_logging(
            'Previously failed to extract organ for feature {' + feature + '} from {' + donor_path + '}')
        logging.warning(red('Previously failed') + f' to extract organ for feature {feature} from {donor_path}')
        # os.remove(failure_test)
        return None

    if os.path.exists(pickle_location):
        utils.perform_logging('Already extracted organ for feature {' + feature + '} from {' + donor_path + '}')
        logging.warning(green('Already extracted') + f' organ for feature {feature} from {donor_path}')
        with open(pickle_location, 'rb') as f:
            return pickle.load(f)

    def failure_occurred(donor_id):
        with open(failure_test, 'wt'):
            pass
        utils.perform_logging('Organ harvest from donor ' + str(donor_id) + ' was failed.')
        logging.warning('Organ harvest from donor ' + str(donor_id) + ' was failed.')
        # print("Feature: ", feature)
        logging.info('Extraction time: {}'.format(
            utils.seconds_to_time(timer() - start)))

    start = timer()
    feature_type, j_feature = drebin.to_j_feature(feature)

    utils.perform_logging('Extracting {' + j_feature + '} from {' + donor_path + '}...')
    logging.debug(yellow(f'Extracting {j_feature} from {donor_path}...'))

    if not os.path.isfile(donor_path):
        utils.perform_logging('Donor app not found: {' + donor_path + '}')
        logging.warning(red(f'Donor app not found: {donor_path}'))
        return None

        # Run the extractor!
    try:
        out = extract(donor_path, j_feature, feature_type)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        failure_occurred(donor_id)
        return None

    if out == "":
        # print("Feature: ", feature)
        return None
    outtemp = out

    out = out.split('\n')
    try:
        if 'Dependencies exported and slice' in out[-2]:
            utils.perform_logging('Organ harvest from donor ' + str(+donor_id) + ' was successful!')
            logging.info('Organ harvesting from donor' + str(donor_id) + ' was successful!')
        elif 'Dependencies exported but no slice' in out[-2]:
            utils.perform_logging('Organ harvest from donor ' + str(donor_id) + ' was successful, but needs vein')
            logging.info('Organ harvest from donor ' + str(donor_id) + ' was successful, but needs vein')
            organ.needs_vein = True
        else:
            utils.perform_logging(out[-3:])
            logging.info(out[-3:])

            if os.path.exists(failure_test) == False:
                os.makedirs(organ.location, exist_ok=True)
            utils.perform_logging('Organ harvest from donor ' + str(donor_id) + ' was failed')
            failure_occurred(donor_id)
            return None
    except:
        out = outtemp.split('\\n')
        if 'Dependencies exported and slice' in out[-2]:
            utils.perform_logging('Organ harvest from donor ' + str(donor_id) + ' was successful!')
            logging.info('Organ harvest from donor ' + str(donor_id) + ' was successful!')
            # print("Organ harvest successful! feature: " , j_feature)
        elif 'Dependencies exported but no slice' in out[-2]:
            utils.perform_logging('Organ harvest from donor ' + str(donor_id) + ' was successful, but needs vein')
            logging.info('Organ harvest from donor ' + str(donor_id) + ' was successful, but needs vein')
            # print("Organ harvest successful, but needs vein! feature: " , j_feature)
            organ.needs_vein = True
        else:
            logging.info(out[-3:])
            utils.perform_logging('Organ harvest from donor ' + str(donor_id) + ' was failed')
            failure_occurred(donor_id)
            # print("Feature: ", feature)
            return None

            # Get the list of classes referenced/contained by the slice
    classes_list = os.path.join(organ.location, 'classes.txt')
    with open(classes_list, 'r') as f:
        organ.classes = {x.strip() for x in f.readlines()}

    # Weigh the organ
    logging.debug(f'Evaluating feature {organ.feature}')

    operating_room = tempfile.mkdtemp(dir=evadedroid_config['tmp_dir'])
    template = os.path.join(evadedroid_config['template_path'], 'template.apk')
    template = shutil.copy(template, operating_room)

    logging.debug(blue('Calling the injector...'))
    utils.perform_logging('Calling the injector...')
    out = utils.run_java_component(evadedroid_config['template_injector'],
                                   [template,
                                    organ.location,
                                    evadedroid_config['android_sdk']])

    if out == "":
        print("Injection from donor " + str(donor_id) + " was failed")
        utils.perform_logging("Injection from donor " + str(donor_id) + " was failed")
        shutil.rmtree(operating_room)
        return None

    if out.find("Injection done") == -1:
        failure_occurred(donor_id)
        shutil.rmtree(operating_room)
        print("Injection from donor " + str(donor_id) + " was failed")
        utils.perform_logging("Injection from donor " + str(donor_id) + " was failed")
        return None

    utils.perform_logging("Injection to template completed successfully")
    logging.debug("Injection to template completed successfully")
    post_op = os.path.join(operating_room, 'sootOutput', 'template.apk')

    try:
        organ.feature_dict = drebin.get_features(post_op)
    except:
        utils.perform_logging("feature extraction was failed")
        return None

    try:
        organ.feature_dict.pop("activities::_TemplateMainActivity")
        organ.feature_dict.pop("activities::template_template_TemplateMainActivity")
    except:
        organ.feature_dict.pop("activities::b'_TemplateMainActivity'")
        organ.feature_dict.pop("activities::b'template_template_TemplateMainActivity'")

    organ.number_of_side_effect_features = len(organ.feature_dict) - 1

    count_permissions(organ)
    organ.extraction_time = timer() - start
    logging.info('Extraction time: {}'.format(
        utils.seconds_to_time(organ.extraction_time)))

    with open(pickle_location, 'wb') as f:
        pickle.dump(organ, f)
    shutil.rmtree(operating_room)
    return organ


# This function that is originally created in [1] has been modified for using in EvadeDroid
def extract(apk, j_feature, feature_type):
    """Extract feature from given donor apk."""

    if feature_type == "api_permissions":
        feature_type = "APIPermission"
    elif feature_type == "api_calls" or feature_type == "interesting_calls":
        feature_type = "APICall"
    else:
        return

    try:
        out = subprocess.check_output(
            [evadedroid_config['java_sdk'] + 'java', '-jar', evadedroid_config['extractor'], j_feature,
             apk, feature_type, evadedroid_config['ice_box'],
             evadedroid_config['android_sdk']], stderr=subprocess.PIPE,
            timeout=evadedroid_config['extractor_timeout'])
        try:
            out = str(out, 'utf-8')
        except:
            out = str(out)
            out = out.replace("xa3", "a")
            out = out.replace("b'", "")
            out = out.replace("'", "")
            out = bytes(out, 'utf-8')
            out = str(out, 'utf-8')
    except subprocess.TimeoutExpired:
        logging.debug(f'Extractor timed out during {apk}, skipping feature')
        raise
    except subprocess.CalledProcessError as e:
        try:
            exception = "\nexit code :{0} \nSTDOUT :{1} \nSTDERROR : {2} ".format(
                e.returncode,
                e.output.decode(sys.getfilesystemencoding()),
                e.stderr.decode(sys.getfilesystemencoding()))
        except:
            exception = "\nexit code :{0} \nSTDOUT :{1} \nSTDERROR : {2} ".format(
                e.returncode,
                e.output,
                e.stderr)
        logfile = 'extraction-exception-{}-{}.log'.format(
            str(uuid.uuid4())[:8],
            datetime.strftime(datetime.now(), '%m-%d--%H:%M'))
        logfile = os.path.join('logs', logfile)
        raise exception
    return out


# This function is originally created in [1]
def count_permissions(organ):
    """Count the permissions present in an organ."""
    for feature in organ.feature_dict:
        if "android_permission" in feature:
            organ.permissions.add(feature)
            splits = feature.split("::")[1].replace("_", ".").split(".")
            if len(splits) == 3:
                tmp_p = splits[2]
            elif len(splits) == 4:
                tmp_p = splits[2] + "_" + splits[3]
            elif len(splits) == 5:
                tmp_p = splits[2] + "_" + splits[3] + "_" + splits[4]
            try:
                if not tmp_p or tmp_p in utils.dangerous_permissions:
                    organ.dangerous_permissions = True
            except:
                organ.dangerous_permissions = False


"""
[1] Intriguing Properties of Adversarial ML Attacks in the Problem Space 
    [S&P 2020], Pierazzi et al.
"""
