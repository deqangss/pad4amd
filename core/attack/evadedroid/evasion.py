"""
Generating a real-world adversarial example from an Android APK
"""
from timeit import default_timer as timer
import numpy as np
import os
import shutil
import glob
import random
import pickle

from core.attack.evadedroid import evadedroid_sample as Sample
from core.attack.evadedroid.program_slicing.transformation import inpatients as inpatients
from core.attack.evadedroid.program_slicing.transformation import injection as injection
from core.attack.evadedroid.drebin import drebin as drebin
from core.attack.evadedroid import utils
from core.attack.evadedroid.utils import green, blue
from core.attack.evadedroid.settings import config as evadedroid_config
from config import logging
from core.attack.evadedroid.android_malware_with_n_gram import batch_disasseble, bytecode_extract, n_gram
from core.droidfeature.feature_extraction import Apk2features, feature_gen

import torch
import torch.nn.functional as F


def loss_function(model, x_malware, x_manipulated, malware_detector):
    if malware_detector == "MaMaDroid":
        y_scores_malware = model.clf.predict_proba(x_malware)
        y_scores_adv_malware = model.clf.predict_proba(x_manipulated)
        y_scores_malware = y_scores_malware[0][0]
        y_scores_adv_malware = y_scores_adv_malware[0][0]
        print("y_scores_malware: ", y_scores_malware)
        print("y_scores_adv_malware: ", y_scores_adv_malware)
        # It would be better to consider only y_scores_adv_malware because y_scores_malware is constant
        loss = y_scores_adv_malware - y_scores_malware
        y_pred_adv = model.clf.predict(x_manipulated)[0]
    elif malware_detector == 'AdversarialDeepEnsembleMax':
        y_scores_adv_malware = model.test_new(x_manipulated, [1], 'proba')[0, 0]
        loss = y_scores_adv_malware
        if y_scores_adv_malware > 0.5:
            y_pred_adv = 0
        else:
            y_pred_adv = 1
    elif 'pad4amd' in malware_detector:
        prob_g = None
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(x_manipulated)
        else:
            logits_f = model.forward(x_manipulated)
        label = torch.ones([1, ], dtype=torch.long, device=model.device)
        loss = F.cross_entropy(logits_f, label, reduction='none')
        y_pred = logits_f.argmax(1)
        y_pred_adv = y_pred[0].item()
        print('clf loss:', loss, logits_f)
        if hasattr(model, 'is_detector_enabled'):
            tau = model.get_tau_sample_wise(y_pred)
            loss = loss + (torch.clamp(tau - prob_g, max=1.0))
            print('de loss:', torch.clamp(tau - prob_g, max=1.0))
            indicator = prob_g <= tau
            y_pred_adv = int(y_pred_adv | ((~indicator)[0].item()))
    else:
        y_scores_malware = model.clf.decision_function(x_malware)
        y_scores_adv_malware = model.clf.decision_function(x_manipulated)
        loss = y_scores_malware - y_scores_adv_malware
        y_pred_adv = model.clf.predict(x_manipulated)[0]
    return loss, y_pred_adv


def loss_function_for_hard_label(model, post_op_host, x_manipulated, malware_detector):
    rootdir = evadedroid_config['tmp_dir'] + "smalis"
    res, fullTopath = batch_disasseble.disassemble_adv(post_op_host, rootdir, 3000)
    if res == 0:
        bytecode_extract.collect_adv(fullTopath, 1, post_op_host)
        loss = n_gram.extract_n_gram_adv(5, post_op_host)
        print("hard-label-loss: " + str(loss))
        loss = utils.round_down(loss, 1)
        print("hard-label-loss after round down: " + str(loss))
    else:
        loss = 0

    print("post_op_host: ", post_op_host)
    if malware_detector == "Drebin":
        y_pred_adv = model.clf.predict(x_manipulated)[0]
    # elif malware_detector == "Kaspersky":
    #     no_detect, Kaspersky, _, _, _, _, _, _, _, _, _ = vt.report(post_op_host)
    #     y_pred_adv = bool(int(Kaspersky))
    # elif malware_detector == "McAfee":
    #     no_detect, _, _, McAfee, _, _, _, _, _, _, _ = vt.report(post_op_host)
    #     y_pred_adv = bool(int(McAfee))
    # elif malware_detector == "Avira":
    #     no_detect, _, _, _, _, Avira, _, _, _, _, _ = vt.report(post_op_host)
    #     y_pred_adv = bool(int(Avira))
    # elif malware_detector == "Ikarus":
    #     no_detect, _, _, _, _, _, _, Ikarus, _, _, _ = vt.report(post_op_host)
    #     y_pred_adv = bool(int(Ikarus))
    # elif malware_detector == "BitDefenderFalx":
    #     no_detect, _, _, _, _, _, _, _, _, BitDefenderFalx, _ = vt.report(post_op_host)
    #     y_pred_adv = bool(int(BitDefenderFalx))
    elif 'pad4amd' in malware_detector:
        if hasattr(model, 'is_detector_enabled'):
            logits_f, prob_g = model.forward(x_manipulated)
        else:
            logits_f = model.forward(x_manipulated)

        y_pred_ = logits_f.argmax(1)
        y_pred = y_pred_[0].item()
        if hasattr(model, 'is_detector_enabled'):
            tau = model.get_tau_sample_wise(y_pred_)
            indicator = prob_g <= tau
            y_pred = int(y_pred | ((~indicator)[0].item()))
    return loss, y_pred, 0


def generate_adversarial_example(malware, action_set, q, increase_in_size,
                                 model_inaccessible, hard_label, malware_detector='pad4amd'):
    query_time = 0
    k = 1
    app_name = os.path.basename(malware)
    percentage_increasing_size = 0
    number_of_features_adv_malware_per_query = list()
    number_of_api_calls_adv_malware_per_query = list()
    transformations = list()
    M_total = []
    increase_in_size_current = 0

    number_of_features_adv_malware = 0
    number_of_api_calls_adv_malware = 0

    print("malware: " + str(malware))
    utils.perform_logging_for_attack("malware: " + str(malware))
    logging.info(blue('Loading host malware...'))
    host = inpatients.Host.load(malware)
    os.makedirs(host.tmpdname, exist_ok=True)
    host_path = os.path.join(host.tmpdname, host.name)
    if not os.path.exists(host_path):
        shutil.copyfile(malware, host_path)
    logging.info(green(f'Host {host.name} loaded!'))

    y_pred_adv = 1
    sampling_distribution = list(action_set.keys())
    result = 1

    U = list(range(0, len(sampling_distribution)))
    M = []
    malware_dict = drebin.get_features(malware)

    no_malware_feature = len(malware_dict.keys())
    number_of_features_malware = no_malware_feature
    no_of_api_calls_malware = len([f for f in malware_dict.keys() if 'api_calls' in f or 'interesting_calls' in f])
    number_of_api_calls_malware = no_of_api_calls_malware

    print("number of malware features: " + str(no_malware_feature))
    utils.perform_logging_for_attack("number of malware features: " + str(no_malware_feature))
    no_adv_malware_feature = no_malware_feature
    number_of_features_adv_malware = no_adv_malware_feature

    feature_malware_keys = malware_dict.keys()
    feature_malware_no_api_call = [f for f in feature_malware_keys if 'api_calls' in f]
    print("no api calls in malware: " + str(len(feature_malware_no_api_call)))
    utils.perform_logging_for_attack("no api calls in malware: " + str(len(feature_malware_no_api_call)))

    malware_size = os.path.getsize(malware)
    print("malware size (byte): " + str(malware_size))
    utils.perform_logging_for_attack("malware size (byte): " + str(malware_size))

    if malware_detector == "MaMaDroid":
        # path = os.path.join(evadedroid_config['mamadroid'], 'Features/Families/dataset_accessible_malware.p')
        # with open(path, 'rb') as f:
        #     apks_path_for_mamadroid = pickle.load(f)
        # apks_path_for_mamadroid.pop(0)
        #
        # # x_malware_mamadroid = [item[1:] for item in apks_path_for_mamadroid if item[0].replace('.txt','.apk') == os.path.basename(malware)]
        # x_malware_mamadroid = [item[1:] for item in apks_path_for_mamadroid if
        #                        os.path.splitext(item[0])[0] + '.apk' == os.path.basename(malware)]
        # print("x_malware_mamadroid: ", str(x_malware_mamadroid[0]))
        # x_malware_mamadroid = x_malware_mamadroid[0]
        # x_malware_mamadroid = np.array(x_malware_mamadroid)
        # x_malware_mamadroid = x_malware_mamadroid.reshape(1, -1)
        # y_pred_adv = model_inaccessible.clf.predict(x_malware_mamadroid)[0]
        # decision_score = model_inaccessible.clf.predict_proba(x_malware_mamadroid)
        # decision_score = decision_score[0][0]
        new_adv_dict = malware_dict
        L_best = 0  # model_inaccessible.clf.decision_function(x_malware)
        pass
    elif malware_detector == "Drebin" or malware_detector == "SecSVM":
        # x_malware = model_inaccessible.dict_to_feature_vector(malware_dict)
        # y_pred_adv = model_inaccessible.clf.predict(x_malware)[0]
        # decision_score = model_inaccessible.clf.decision_function(x_malware)
        new_adv_dict = malware_dict
        L_best = -0.01  # model_inaccessible.clf.decision_function(x_malware)
        pass
    elif malware_detector == 'AdversarialDeepEnsembleMax':
        # x_malware = model_inaccessible.dict_to_feature_vector(malware_dict)
        # y_pred_adv = model_inaccessible.test_new(x_malware, [1], 'label')[0]
        # decision_score = model_inaccessible.test_new(x_malware, [1], 'proba')[0, 0]
        new_adv_dict = malware_dict
        L_best = -0.01  # model_inaccessible.clf.decision_function(x_malware)
        pass
    elif 'pad4amd' in malware_detector:
        feature_extractor = Apk2features(evadedroid_config['feature_pool'],
                                         evadedroid_config['data_intermediate'])
        feature_file_path = evadedroid_config['feature_pool'] + os.path.splitext(os.path.basename(malware))[0] + '.feat'
        if not os.path.exists(feature_file_path):
            raise FileNotFoundError("Cannot find the path {}.".format(feature_file_path))
        vocab, _1, _2 = feature_extractor.get_vocab()
        x_malware, _1 = feature_extractor.feature2ipt(feature_file_path, label=1, vocabulary=vocab)
        x_malware = torch.from_numpy(x_malware.reshape([1, -1])).to(model_inaccessible.device).double()
        y_cent, x_density = model_inaccessible.inference_batch_wise(x_malware)
        y_pred = y_cent.argmax(1)
        y_pred_adv = y_pred[0].item()
        if 'indicator' in type(model_inaccessible).__dict__.keys():
            indicator_flag = model_inaccessible.indicator(x_density, y_pred)
            y_pred_adv = int(y_pred_adv | ((~indicator_flag)[0].item()))
        new_adv_dict = malware_dict
        L_best = -0.01
    else:
        new_adv_dict = dict()
        # if malware_detector == "Kaspersky":
        #     no_detect_best, Kaspersky, _, _, _, _, _, _, _, _, _ = vt.report(malware)
        #     y_pred_adv = bool(int(Kaspersky))
        # elif malware_detector == "McAfee":
        #     no_detect_best, _, _, McAfee, _, _, _, _, _, _, _ = vt.report(malware)
        #     y_pred_adv = bool(int(McAfee))
        # elif malware_detector == "Avira":
        #     no_detect_best, _, _, _, _, Avira, _, _, _, _, _ = vt.report(malware)
        #     y_pred_adv = bool(int(Avira))
        # elif malware_detector == "Ikarus":
        #     no_detect_best, _, _, _, _, _, _, Ikarus, _, _, _ = vt.report(malware)
        #     y_pred_adv = bool(int(Ikarus))
        # elif malware_detector == "BitDefenderFalx":
        #     no_detect_best, _, _, _, _, _, _, _, _, BitDefenderFalx, _ = vt.report(malware)
        #     y_pred_adv = bool(int(BitDefenderFalx))
        L_best = 0
        pass

    start = timer()
    number_of_query = 0

    print("label malware: " + str(y_pred_adv))
    malware_label = y_pred_adv
    utils.perform_logging_for_attack("label malware: " + str(y_pred_adv))
    is_intact = 1
    is_try_to_inject = 0  # This flag is used to show that at least one injection was done duting transplantation

    label_per_query = dict()
    modified_features_per_query = dict()

    cnt_size_check = 0
    cnt_injection_failed = 0
    post_op_host = os.path.join(os.path.join(host.tmpdname, 'postop'), host.name)
    os.makedirs(os.path.dirname(post_op_host), exist_ok=True)
    #  shutil.copy(malware, post_op_host)

    while number_of_query < q and y_pred_adv == 1:

        if cnt_size_check > 5 or cnt_injection_failed > 5:
            break
        U = [x for x in U if
             x not in M]  # We should remove the features that have been already modified. Note they also includes side effect features
        result = 1

        print("len(U): " + str(len(U)))
        if len(U) == 0:
            break
        if len(U) > len(M):
            if len(U) > k:
                M = random.sample(U, k)
            else:
                M = U
        else:
            M = U

        for m in range(0, len(M)):
            M_total.append(M[m])
        apks = []
        for i in range(0, len(M_total)):
            # print("Tansformation No: ",i)
            if action_set.get(sampling_distribution[M_total[i]]) == None:
                continue
            organ = action_set[sampling_distribution[M_total[i]]]
            apks.append(organ.location)

        utils.perform_logging_for_attack("sampling: " + str(M_total[i]))
        print("sampling - No: " + str(M_total[i]))
        print("sampling - Feature: " + str(sampling_distribution[M_total[i]]))

        utils.perform_logging_for_attack(
            "start tranfromation - no of query: %d - app: %s" % (number_of_query, app_name))
        print("start tranfromation - no of query: %d - app: %s" % (number_of_query, app_name))
        result, post_op_host, side_effects = injection.transplant_organs(host, apks)
        print("end tranfromation - no of query: %d - app: %s - result: %d - cnt_injection_failed: %d" % (
            number_of_query, app_name, result, cnt_injection_failed))
        utils.perform_logging_for_attack("end tranfromation - no of query: %d - app: %s" % (number_of_query, app_name))

        if result == 1:
            cnt_injection_failed += 1
            # modified_features_per_query[number_of_query] = no_adv_malware_feature - no_malware_feature
            for m in range(0, len(M)):
                M_total.remove(M[m])
            continue
        cnt_injection_failed = 0  # reset it one one organ inject successfuly
        is_try_to_inject = 1  # This flag show that at least one injection was done
        new_adv_dict_temp = new_adv_dict
        if malware_detector == "MaMaDroid" or malware_detector == "Drebin" or malware_detector == "SecSVM" or \
                malware_detector == 'AdversarialDeepEnsembleMax' or ('pad4amd' in malware_detector):
            new_adv_dict = drebin.get_features(post_op_host)
        else:
            new_adv_dict = dict()
        no_adv_malware_feature = len(new_adv_dict.keys())
        no_of_api_calls_adv_malware = len(
            [f for f in new_adv_dict.keys() if 'api_calls' in f or 'interesting_calls' in f])

        print("number of adv malware features: " + str(no_adv_malware_feature))
        utils.perform_logging_for_attack("number of adv malware features: " + str(no_adv_malware_feature))

        feature_adv_malware_keys = new_adv_dict.keys()
        feature_adv_malware_no_api_call = [f for f in feature_adv_malware_keys if 'api_calls' in f]
        print("no api calls in malware: " + str(len(feature_adv_malware_no_api_call)))
        utils.perform_logging_for_attack("no api calls in malware: " + str(len(feature_adv_malware_no_api_call)))

        adv_malware_size = os.path.getsize(post_op_host)
        print("adv_malware size (byte): " + str(adv_malware_size))
        utils.perform_logging_for_attack("adv_malware size (byte): " + str(adv_malware_size))

        increase_in_size_current = (adv_malware_size - malware_size) / malware_size
        print("increase_in_size_current size (%): " + str(increase_in_size_current))
        utils.perform_logging_for_attack("increase_in_size_current size (%): " + str(increase_in_size_current))

        print("cnt_size_check: %s - Check increase size: %s" % (
            cnt_size_check, str(float(increase_in_size_current) <= float(increase_in_size))))
        if (float(increase_in_size_current) > 0 and float(increase_in_size_current) <= float(increase_in_size)):
            # Any way we should consider all features
            # new_adv_dict = soot_filter(host.features, new_adv_dict, side_effects)
            cnt_size_check = 0  # reset it once the size is ok
            if malware_detector == "MaMaDroid":
                # try:
                #     # db = "sample_" + os.path.basename(malware).replace('.apk','').replace('.','_')
                #     db = "sample_" + os.path.splitext(os.path.basename(malware))[0].replace('.', '_')
                #     mamadroid.api_sequence_extraction([post_op_host], db)
                #     _app_dir = evadedroid_config['mamadroid']
                #     no_finished_apps = len(os.listdir(os.path.join(_app_dir, 'graphs', db)))
                #     if no_finished_apps == 0:
                #         print("Remove transformation because of failing in creating api call graph")
                #         files = glob.glob(host.tmpdname + "/postop/*")
                #         for f in files:
                #             os.remove(f)
                #         for m in range(0, len(M)):
                #             M_total.remove(M[m])
                #         continue
                #     dbs = list()
                #     dbs.append(db)
                #     MaMaStat.feature_extraction_markov_chain(dbs)
                #     path = os.path.join(evadedroid_config['mamadroid'], "Features/Families/" + db + ".p")
                #     with open(path, 'rb') as f:
                #         apks_path_for_mamadroid = pickle.load(f)
                #     apks_path_for_mamadroid.pop(0)
                #     x_manipulated_mamadroid = [item[1:] for item in apks_path_for_mamadroid]
                #     print("x_manipulated_mamadroid: ", str(x_manipulated_mamadroid[0]))
                #     x_manipulated_mamadroid = x_manipulated_mamadroid[0]
                #     x_manipulated_mamadroid = np.array(x_manipulated_mamadroid)
                #     x_manipulated_mamadroid = x_manipulated_mamadroid.reshape(1, -1)
                #     _app_dir = evadedroid_config['mamadroid']
                #     os.remove(_app_dir + "/Features/Families/" + db + '.p')
                #     shutil.rmtree(_app_dir + "/graphs/" + db)
                #     shutil.rmtree(_app_dir + "/package/" + db)
                #     shutil.rmtree(_app_dir + "/family/" + db)
                #     shutil.rmtree(_app_dir + "/class/" + db)
                #
                #     # shutil.rmtree(_app_dir + "/Features/Packages" + db +'.p')
                # except Exception as e:
                #     print("exception: ", e)
                #     x_manipulated_mamadroid = x_malware_mamadroid
                pass
            elif malware_detector == "Drebin" or malware_detector == "SecSVM" or malware_detector == 'AdversarialDeepEnsembleMax':
                x_manipulated = model_inaccessible.dict_to_feature_vector(new_adv_dict)
            elif 'pad4amd' in malware_detector:
                sp = os.path.join(os.path.dirname(post_op_host),
                                  os.path.splitext(os.path.basename(post_op_host))[0] + feature_extractor.file_ext)
                feature_gen.apk2features(post_op_host, saving_path=sp)
                x_manipulated, _1 = feature_extractor.feature2ipt(sp, label=1, vocabulary=vocab)
                x_manipulated = torch.from_numpy(x_manipulated.reshape([1, -1])).to(model_inaccessible.device).double()
            else:
                x_manipulated = ""

            if hard_label == False:
                if malware_detector != "MaMaDroid":
                    L, y_pred_adv = loss_function(model_inaccessible, x_malware, x_manipulated, malware_detector)
                elif malware_detector == "MaMaDroid":
                    # L, y_pred_adv = loss_function(model_inaccessible, x_malware_mamadroid, x_manipulated_mamadroid,
                    #                               malware_detector)
                    pass
                elif 'pad4amd' in malware_detector:
                    L, y_pred_adv = loss_function(model_inaccessible, x_malware, x_manipulated, malware_detector)
            else:
                start_query = timer()
                L, y_pred_adv, no_detect = loss_function_for_hard_label(model_inaccessible, post_op_host, x_manipulated,
                                                                        malware_detector)
                end_query = timer()
                query_time += end_query - start_query

            if malware_detector != "Drebin" and malware_detector != "SecSVM" and malware_detector != "MaMaDroid" and malware_detector != 'AdversarialDeepEnsembleMax' and 'pad4amd' not in malware_detector:
                if no_detect > no_detect_best:
                    print("no_detect > no_detect_previou: True")
                    L = 0
                else:
                    no_detect_best = no_detect

            print("current loss: " + str(L))
            utils.perform_logging_for_attack("current loss: " + str(L))
            number_of_query += 1

            label_per_query[number_of_query] = y_pred_adv

            if hard_label == False:
                if malware_detector != "MaMaDroid":
                    loss_cmp = L > L_best
                else:
                    loss_cmp = L >= L_best
            else:
                loss_cmp = L >= L_best

            print("loss_cmp: " + str(loss_cmp))
            if loss_cmp:
                modified_features_per_query[number_of_query] = no_adv_malware_feature - no_malware_feature
                is_intact = 0
                L_best = L
                number_of_features_adv_malware_per_query.append(no_adv_malware_feature)
                number_of_api_calls_adv_malware_per_query.append(no_of_api_calls_adv_malware)
            else:
                if number_of_query == 1:
                    modified_features_per_query[number_of_query] = no_adv_malware_feature - no_malware_feature

                    number_of_features_adv_malware_per_query.append(no_malware_feature)
                    number_of_api_calls_adv_malware_per_query.append(no_of_api_calls_malware)

                else:
                    modified_features_per_query[number_of_query] = modified_features_per_query[number_of_query - 1]

                    length = len(number_of_features_adv_malware_per_query)
                    number_of_features_adv_malware_per_query.append(
                        number_of_features_adv_malware_per_query[length - 1])
                    length = len(number_of_api_calls_adv_malware_per_query)
                    number_of_api_calls_adv_malware_per_query.append(
                        number_of_api_calls_adv_malware_per_query[length - 1])
                for m in range(0, len(M)):
                    M_total.remove(M[m])

            files = glob.glob(host.tmpdname + "/postop/*")
            if y_pred_adv == 1:
                for f in files:
                    os.remove(f)

            utils.perform_logging_for_attack("number of query: " + str(number_of_query) +
                                             " - current loss: " + str(L) + " - best loss: " + str(
                L_best) + " - best actions: " + str(M_total))
            print("number of query: " + str(number_of_query) +
                  " - current loss: " + str(L) + " - best loss: " + str(L_best) + " - best actions: " + str(M_total))
        else:
            cnt_size_check += 1
            new_adv_dict = new_adv_dict_temp  # roll back new_adv_dict
            files = glob.glob(host.tmpdname + "/postop/*")
            for f in files:
                os.remove(f)
            for m in range(0, len(M)):
                M_total.remove(M[m])

    if malware_detector == "Drebin" or malware_detector == "SecSVM":
        if is_intact == 0:
            adv_malware_label = model_inaccessible.clf.predict(x_manipulated)[0]
            adv_decision_score = model_inaccessible.clf.decision_function(x_manipulated)[0]
        else:
            adv_malware_label = model_inaccessible.clf.predict(x_malware)[0]
            adv_decision_score = model_inaccessible.clf.decision_function(x_malware)[0]
    elif malware_detector == "MaMaDroid":
        if is_intact == 0:
            adv_malware_label = model_inaccessible.clf.predict(x_manipulated_mamadroid)[0]
            adv_decision_score = model_inaccessible.clf.predict_proba(x_manipulated_mamadroid)
        else:
            adv_malware_label = model_inaccessible.clf.predict(x_malware_mamadroid)[0]
            adv_decision_score = model_inaccessible.clf.predict_proba(x_malware_mamadroid)
    elif malware_detector == 'AdversarialDeepEnsembleMax':
        if is_intact == 0:
            adv_malware_label = model_inaccessible.test_new(x_manipulated, [1], 'label')[0]
            adv_decision_score = model_inaccessible.test_new(x_manipulated, [1], 'proba')[0, 0]
        else:
            adv_malware_label = model_inaccessible.test_new(x_malware, [1], 'label')[0]
            adv_decision_score = model_inaccessible.test_new(x_malware, [1], 'proba')[0, 0]
    elif 'pad4amd' in malware_detector:
        if is_intact == 0:
            y_cent, x_density = model_inaccessible.inference_batch_wise(x_manipulated)
            y_pred = y_cent.argmax(1)
            adv_decision_score = y_cent[0, y_pred[0]]
            adv_malware_label = y_pred[0].item()
            if 'indicator' in type(model_inaccessible).__dict__.keys():
                indicator_flag = model_inaccessible.indicator(x_density, y_pred)
                adv_malware_label = int(adv_malware_label | (~indicator_flag)[0].item())
        else:
            y_cent, x_density = model_inaccessible.inference_batch_wise(x_malware)
            y_pred = y_cent.argmax(1)
            adv_decision_score = y_cent[0, y_pred[0]]
            adv_malware_label = y_pred[0].item()
            if 'indicator' in type(model_inaccessible).__dict__.keys():
                indicator_flag = model_inaccessible.indicator(x_density, y_pred)
                adv_malware_label = int(adv_malware_label | (~indicator_flag)[0].item())
    else:
        if is_intact == 0:
            if os.path.exists(post_op_host) == False:
                adv_malware_label = 1
            else:
                start_query = timer()
                if malware_detector == "Kaspersky":
                    _, Kaspersky, _, _, _, _, _, _, _, _, _ = vt.report(post_op_host)
                    adv_malware_label = bool(int(Kaspersky))
                elif malware_detector == "McAfee":
                    _, _, _, McAfee, _, _, _, _, _, _, _ = vt.report(post_op_host)
                    adv_malware_label = bool(int(McAfee))
                elif malware_detector == "Avira":
                    _, _, _, _, _, Avira, _, _, _, _, _ = vt.report(post_op_host)
                    adv_malware_label = bool(int(Avira))
                elif malware_detector == "Ikarus":
                    _, _, _, _, _, _, _, Ikarus, _, _, _ = vt.report(post_op_host)
                    adv_malware_label = bool(int(Ikarus))
                elif malware_detector == "BitDefenderFalx":
                    _, _, _, _, _, _, _, _, _, BitDefenderFalx, _ = vt.report(post_op_host)
                    adv_malware_label = bool(int(BitDefenderFalx))
                end_query = timer()
                query_time += end_query - start_query
        else:
            adv_malware_label = 1

    fname, fext = os.path.splitext(host.name)
    dest = os.path.join(host.results_dir + "/postop", host.name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if y_pred_adv == 0:
        shutil.copyfile(post_op_host, dest)
    else:
        shutil.copyfile(host_path, dest)
    shutil.rmtree(host.tmpdname)
    number_of_queries = number_of_query
    if adv_malware_label == 0:
        percentage_increasing_size = increase_in_size_current
        transformations = M_total
        number_of_features_adv_malware = no_adv_malware_feature
        number_of_api_calls_adv_malware = no_of_api_calls_adv_malware

    intact_due_to_soot_error = 1 - is_try_to_inject
    end = timer()
    # execution_time = end - start
    execution_time = end - start - query_time
    print("execution_time: ", execution_time)
    print("query_time: ", query_time)
    classified_with_hard_label = hard_label
    apk = Sample.APK(app_name, malware_label, adv_malware_label,
                     number_of_queries, percentage_increasing_size,
                     number_of_features_malware, number_of_features_adv_malware,
                     number_of_features_adv_malware_per_query,
                     number_of_api_calls_malware, number_of_api_calls_adv_malware,
                     number_of_api_calls_adv_malware_per_query, transformations,
                     intact_due_to_soot_error, execution_time, classified_with_hard_label, query_time)

    return apk


def resign(app_path):
    """Resign the apk."""
    utils.run_java_component(evadedroid_config['resigner'], ['--overwrite', '-a', app_path])
