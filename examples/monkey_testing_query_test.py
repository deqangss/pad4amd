import os
import numpy as np
import tempfile
import shutil
import time
import argparse
from core.oracle import APKTestADB

from config import config, logging, ErrorHandler

logger = logging.getLogger('examples.mimicry')
logger.addHandler(ErrorHandler)

monkeyt_argparse = argparse.ArgumentParser(description='arguments for mimicry attack')
monkeyt_argparse.add_argument('--n_samples', type=int, default=20,
                              help='samples for functionality testing.')
monkeyt_argparse.add_argument('--attacks', type=str, default='max,mimicry,gdkde',
                              help="attacks:max, mimicry, gdkde, pgdlinf, etc.")
monkeyt_argparse.add_argument('--models', type=str, default='madvtrain',
                              help="model type, either of 'maldet', 'kde', 'gmm', 'madvtrain', 'padvtrain'.")


def _main():
    np.random.seed(0)
    args = monkeyt_argparse.parse_args()
    attack_names = args.attacks.split(',')
    model_names = args.models.split(',')

    sample_save_dirs = []
    malicious_sample_dir = config.get('dataset', 'malware_dir')

    for attack in attack_names:
        for model in model_names:
            save_dir = os.path.join(
                os.path.join(config.get('experiments', attack), model), 'adv_apps')
            sample_save_dirs.append(save_dir)

    list_of_sets = []
    for save_dir in sample_save_dirs:
        list_of_sets.append(set(os.listdir(save_dir)))
    inter_apps = list(set.intersection(*list_of_sets))

    #
    print("Samples: ", len(inter_apps))
    unperturbed_example_dir = os.path.join(os.path.dirname(malicious_sample_dir), "unpert_examples")
    if not os.path.exists(unperturbed_example_dir):
        os.mkdir(unperturbed_example_dir)

    for app_name in inter_apps:
        app_name_ = app_name.split('_')[0]
        apk_path = os.path.join(malicious_sample_dir, app_name_)
        shutil.copy(apk_path, unperturbed_example_dir)
        for adv_save_dir in sample_save_dirs:
            dup_sample_dir = os.path.join(os.path.dirname(adv_save_dir), 'dup_adv_apps')
            if not os.path.exists(dup_sample_dir):
                os.mkdir(dup_sample_dir)
            shutil.copy(os.path.join(adv_save_dir, app_name), dup_sample_dir)
    return
    #
    n_samples = args.n_samples if len(inter_apps) >= args.n_samples else len(inter_apps)
    app_names = np.random.choice(inter_apps, n_samples, replace=False).tolist()

    apk_test_adb = APKTestADB()
    with tempfile.TemporaryDirectory() as tmpdir:
        for app_name in app_names:
            app_name_ = app_name.split('_')[0]
            apk_path = os.path.join(malicious_sample_dir, app_name_ + '.apk')
            tmp_path = os.path.join(tmpdir, app_name_ + '.apk')
            shutil.copy(apk_path, tmp_path)
            apk_test_adb.submit(tmp_path)
            logger.info("Submit: " + apk_path)
            for save_dir in sample_save_dirs:
                apk_path = os.path.join(save_dir, app_name)
                tmp_path = os.path.join(tmpdir, app_name)
                shutil.copy(apk_path, tmp_path)
                apk_test_adb.submit(tmp_path)
                logger.info("Submit: " + apk_path)

        while True:
            time.sleep(30)

            states = []
            for app_name in app_names:
                app_name_ = app_name.split('_')[0]
                tmp_path = os.path.join(tmpdir, app_name_ + '.apk')
                states.append(apk_test_adb.get_state(tmp_path))
                for save_dir in sample_save_dirs:
                    apk_path = os.path.join(save_dir, app_name)
                    tmp_path = os.path.join(tmpdir, app_name)
                    shutil.copy(apk_path, tmp_path)
                    states.append(apk_test_adb.get_state(tmp_path))

            if all(states):
                break

        org_sample_installed = 0
        install_flag_2dlist = []
        functionality_flag_2dlist = []
        for app_name in app_names:
            app_name_ = app_name.split('_')[0]
            tmp_path = os.path.join(tmpdir, app_name_ + '.apk')
            org_install_flag, org_activities, org_exceptions = apk_test_adb.get_report(tmp_path)
            if not org_install_flag:
                logger.info("Unperturbed example {}: failed to install.".format(app_name_))
                continue
            org_sample_installed += 1
            install_flag_list = []
            func_flag_list = []
            for save_dir in sample_save_dirs:
                apk_path = os.path.join(save_dir, app_name)
                tmp_path = os.path.join(tmpdir, app_name)
                shutil.copy(apk_path, tmp_path)
                adv_install_flag, adv_activities, adv_exceptions = apk_test_adb.get_report(tmp_path)
                install_flag_list.append(adv_install_flag)
                func_flag = (org_activities == adv_activities) & (org_exceptions == adv_exceptions)
                if not func_flag:
                    logger.info("Ruin the functionality: " + apk_path)
                    logger.info('\t Original activities: {}'.format(','.join(list(org_activities))))
                    logger.info('\t Perturbed activities: {}'.format(','.join(list(adv_activities))))
                    logger.info('\t Original exceptions: {}'.format(','.join(list(org_exceptions))))
                    logger.info('\t Perturbed exceptions: {}'.format(','.join(list(org_exceptions))))
                func_flag_list.append(func_flag)

            install_flag_2dlist.append(install_flag_list)
            functionality_flag_2dlist.append(func_flag_list)

        install_count = np.sum(np.array(install_flag_2dlist), axis=0).tolist()
        func_count = np.sum(np.array(functionality_flag_2dlist), axis=0).tolist()
        logger.info("Installable apps: {}.".format(org_sample_installed))
        for i, attack in enumerate(attack_names):
            logger.info("Attack {}: number of installable apks {} and runnable apks {}.".format(attack,
                                                                                                install_count[i],
                                                                                                func_count[i]))

    return


if __name__ == "__main__":
    _main()
