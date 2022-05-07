# -*- coding: utf-8 -*-

"""
Data preparation for the EvadeDroid's pipeline.
"""
import os
from settings import config
from feature_extraction import drebin
import json
import random
import csv
import shutil
import subprocess
import pickle
from mamadroid import mamadroid, MaMaStat
from android_malware_with_n_gram import n_gram, bytecode_extract, batch_disasseble


def get_inputapps(path):    
    malware_apps = os.listdir(path + 'malware')
    normal_apps = os.listdir(path + 'normal')      
    X_dataset = []
    Y_dataset = []
    meta = []
    i = 0
    j = 0
    for app in malware_apps:
        app_path = path + 'malware/' + app
        try:
            feature_set = drebin.get_features(app_path)
        except:
            print("Feature extraction was failed for: ", app)
            continue
        meta_item = {"sha256": feature_set["sha256"] ,"sample_path":app_path}
        meta.append(meta_item)
        feature_set.pop('sha256')  # Ensure hash isn't included in features
        X_dataset.append(feature_set)
        Y_dataset.append(1)     
        i += 1
        j += 1
        print("Number of apps: ", j)
        if i == 1000:
            break
    i = 0   
    for app in normal_apps:
        app_path = path + 'normal/' + app
        try:
            feature_set = drebin.get_features(app_path)
        except:
            print("Feature extraction was failed for: ", app)
            continue
        meta_item = {"sha256": feature_set["sha256"] ,"sample_path":app_path}
        meta.append(meta_item)
        feature_set.pop('sha256')  # Ensure hash isn't included in features
        X_dataset.append(feature_set)
        Y_dataset.append(0) 
        i += 1
        j += 1
        print("Number of apps: ", j)
        if i == 1000:
            break
        
    index = list(range(0,len(X_dataset)))
    random.shuffle(index)
    
    X_dataset_shuffled = [X_dataset[val] for i,val in enumerate(index)]
    Y_dataset_shuffled = [Y_dataset[val] for i,val in enumerate(index)]
    meta_shuffled = [meta[val] for i,val in enumerate(index)]    
    
    
    print("total scan %d apps: %d malwares and %d bengins" % (len(malware_apps) + len(normal_apps), len(malware_apps), len(normal_apps)))  

    return X_dataset_shuffled, Y_dataset_shuffled, meta_shuffled 

# Determine the metadata of 15,000 malware and clean samples from the JSON files 
# of the original dataset that was used in our study. 

def create_sub_dataset():
    sub_dataset_path = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset.p')
    if os.path.exists(sub_dataset_path) == False:
        
        X_filename = os.path.join(config['total_dataset'],"apg-X.json")
        #with open(X_filename, 'rt') as f:
        with open(X_filename, 'rb') as f:
            X = json.load(f)        
        #    [o.pop('sha256') for o in X]  # prune the sha, uncomment if needed    
              
        y_filename = os.path.join(config['total_dataset'],"apg-y.json")
        with open(y_filename, 'rt') as f:
            y = json.load(f)
            #y = [x[0] for x in json.load(f)]  # prune the sha, uncomment if needed    
        
        y_malware = [index for index,item in enumerate(y) if item == 1]
        y_goodware = [index for index,item in enumerate(y) if item == 0]
        
        meta_filename = os.path.join(config['total_dataset'],"apg-meta.json")
        with open(meta_filename, 'rt') as f:
            meta = json.load(f)  
        
        rand_temp = random.sample(range(0,len(y_malware)),3000)
        #sub_dataset_index_malware = [item[index] for index,item in enumerate(y_malware) if index in rand_temp]
        sub_dataset_index_malware = list()
        for index,item in enumerate(y_malware):
            if index in rand_temp:
                sub_dataset_index_malware.append(item)
        
        rand_temp = random.sample(range(0,len(y_goodware)),12000)
        #sub_dataset_index_goodware = [item[index] for index,item in enumerate(y_goodware) if index in rand_temp]
        sub_dataset_index_goodware = list()
        for index,item in enumerate(y_goodware):
           if index in rand_temp:
               sub_dataset_index_goodware.append(item)
        
        sub_dataset_index = sub_dataset_index_malware + sub_dataset_index_goodware
        with open(sub_dataset_path, 'wb') as f:
            pickle.dump(sub_dataset_index,f) 
        
        X_sub_dataset = [item for index,item in enumerate(X) if index in sub_dataset_index]
        Y_sub_dataset = [item for index,item in enumerate(y) if index in sub_dataset_index]
        meta_sub_dataset = [item for index,item in enumerate(meta) if index in sub_dataset_index]
       
        
        with open(config['features']+'sub_dataset/sub_dataset-X.json' , 'w') as f:
           json.dump(X_sub_dataset, f)          
        
        with open(config['features']+'sub_dataset/sub_dataset-Y.json' , 'w') as f:
           json.dump(Y_sub_dataset , f)       
        
        with open(config['features']+'sub_dataset/sub_dataset-meta.json' , 'w') as f:
           json.dump(meta_sub_dataset, f)    

# Prepare a CSV file from the metadata of malware and clean samples.
def create_csv_from_meta_sub_dataset():
    meta_filename = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-meta.json')    
    data_file = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-meta.csv')    
    if os.path.exists(meta_filename) == True and os.path.exists(data_file) == False:
        with open(meta_filename, 'rt') as f:
            meta = json.load(f)          
        data_file = open(data_file, 'w')        
        csv_writer = csv.writer(data_file)
        count = 0       
        for item in meta:            
            if count == 0:
                header = item.keys()
                csv_writer.writerow(header)
                count += 1
            item['dex_date'] = '2021-07-09 00:00:00'
            csv_writer.writerow(item.values())
        data_file.close()

# Move 1000 accessible malware samples (EvadeDroid's malware samples) from AndroZoo reposirtoy to 
# <.config['apks'],'accessible/malware'> after checking their validity (crrectness)        
def check_malware_apks():
    y_filename = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-Y.json')   
    with open(y_filename, 'rt') as f:
        y = json.load(f)
    
    meta_filename = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-meta.json')   
    with open(meta_filename, 'rt') as f:
        meta = json.load(f)  
    
    malware_index = [index for index,item in enumerate(y) if item == 1]
    apps = [item['pkg_name'] + '.apk' for index,item in enumerate(meta) if index in malware_index]
    malware_count = 0  
    cnt = 0
    selected_malware_app = dict()
    path_malware_index = os.path.join(config['features'] , 'sub_dataset/', 'accessible_malware_index.p')
    with open(path_malware_index,'rb') as f:
        selected_malware_app = pickle.load(f)
    selected_malware_app_temp = selected_malware_app.copy()
    malware_count = len(selected_malware_app_temp)
    for app in apps:
        app_path = os.path.join('C:/AndroidDatasets/AndoZoo',app)#os.path.join(config['apks'],'sub_dataset',app)
        if os.path.exists(app_path) == False:
            cnt += 1 
            continue
        if app in selected_malware_app_temp:            
            cnt += 1 
            continue
        malware_size = os.path.getsize(app_path)
        print("cnt: malware_size - " + str(cnt) + ": " + str(malware_size))
        if malware_size >= 5000000:# malware_size >= 4000000 and  malware_size < 5000000:            
            fullTopath = os.path.join(config['tmp_dir'],"smalis",app)
            command = "java -jar " + config['project_root'] + "/android_malware_with_n_gram/apktool.jar d " + app_path + " -o " + fullTopath
            res = subprocess.call(command, shell=True)            
            print("res: " + str(res))
            if res == 0:
                shutil.rmtree(fullTopath)
                command = config['android_sdk'] + '/platform-tools/adb.exe install -t ' + app_path
                install_result = subprocess.call(command, shell=True) 
                print("install_result: " + str(install_result))
                if install_result == 0:
                    #command = config['android_sdk'] + '/platform-tools/adb.exe uninstall ' + app.replace('.apk','')
                    command = config['android_sdk'] + '/platform-tools/adb.exe uninstall ' + os.path.splitext(app)[0]
                    subprocess.call(command, shell=True)                                
                    app_path_des = os.path.join(config['apks'],'accessible/malware',app)
                    shutil.copyfile(app_path, app_path_des)  
                    selected_malware_app[app] = cnt
                    malware_count += 1
                    print("malware_count: ", malware_count) 
                    '''
                    It seems in specified malware indexes are not right. They will determine in determine_smaples_accessible_inaccessible function
                    with open(path_malware_index,'wb') as f:
                        pickle.dump(selected_malware_app,f)
                    '''
                    if malware_count == 1000:
                        break
        cnt += 1  


def determine_smaples_accessible_inaccessible():
    X_filename = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-X.json')   
    with open(X_filename, 'rt') as f:
        X = json.load(f)
        
    y_filename = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-Y.json')   
    with open(y_filename, 'rt') as f:
        Y = json.load(f)
    
    meta_filename = os.path.join(config['features'] , 'sub_dataset/', 'sub_dataset-meta.json')   
    with open(meta_filename, 'rt') as f:
        meta = json.load(f)
    
    goodware_index = [index for index,item in enumerate(Y) if item == 0]
    apps = [item['pkg_name'] + '.apk' for index,item in enumerate(meta) if index in goodware_index] 
    
    rand_temp = random.sample(range(0,len(goodware_index)),2000)
    selected_goodware_app_accessible = dict()
    
   
    selected_goodware_index_accessible = list()
    for index,item in enumerate(goodware_index):
        if index in rand_temp:           
            selected_goodware_app_accessible[apps[index]] = item # note item shows the index in total samples
            selected_goodware_index_accessible.append(item)
    path = os.path.join(config['features'] , 'sub_dataset/', 'accessible_goodware_index.p')
    with open(path, 'wb') as f:
        pickle.dump(selected_goodware_app_accessible,f)
    
    #selected_goodware_index_accessible = selected_goodware_app_accessible.values();    
    print("Start moving smaples to accessible/normal directory")
    apps_temp = selected_goodware_app_accessible.keys()
    cp = 0
    for apk in apps_temp:
        app_path_src = os.path.join('C:/AndroidDatasets/AndoZoo',apk)
        #app_path_src = os.path.join('C:/GitLab/end-to-end_black-box_evasion_attack/data/apks/inaccessible/normal-temp',apk)
        app_path_des = os.path.join(config['apks'],'accessible/normal',apk)
        if os.path.exists(app_path_src) == True:
            shutil.copyfile(app_path_src, app_path_des)  
            os.remove(app_path_src)
            cp += 1
            print("cp - accessible/normal: " + str(cp))
    print("Finish moving smaples to accessible/normal directory")   
    
    
    selected_goodware_index_inaccessible = [item for item in goodware_index if item not in selected_goodware_index_accessible]
    
    selected_goodware_app_inaccessible = dict()
    
    for index,item in enumerate(goodware_index):
        if item in selected_goodware_index_inaccessible:
            selected_goodware_app_inaccessible[apps[index]] = item
    path = os.path.join(config['features'] , 'sub_dataset/', 'inaccessible_goodware_index.p')
    with open(path, 'wb') as f:
        pickle.dump(selected_goodware_app_inaccessible,f)    
    
    print("Start moving smaples to inaccessible/normal directory")
    apps_temp = selected_goodware_app_inaccessible.keys()
    cp = 0
    for apk in apps_temp:
        app_path_src = os.path.join('C:/AndroidDatasets/AndoZoo',apk)
        #app_path_src = os.path.join('C:/GitLab/end-to-end_black-box_evasion_attack/data/apks/inaccessible/normal-temp',apk)
        app_path_des = os.path.join(config['apks'],'inaccessible/normal',apk)
        if os.path.exists(app_path_src) == True:
            shutil.copyfile(app_path_src, app_path_des)  
            os.remove(app_path_src)
            cp += 1
            print("cp - inaccessible/normal: " + str(cp))
    print("Finish moving smaples to inaccessible/normal directory")
    
    malware_index = [index for index,item in enumerate(Y) if item == 1] 
    apps_malware_total = [item['pkg_name'] + '.apk' for index,item in enumerate(meta) if index in malware_index]
    
    #correct selected_malware_app which is determined in check_malware_apks   
    path = os.path.join(config['apks'],'accessible/malware')
    apps_malware = os.listdir(path)
    apps_malware_index = [index for index,item in enumerate(meta) if item['pkg_name'] + '.apk' in apps_malware]
    selected_malware_app = dict()
    for app in apps_malware:
        apps_malware_index = [index for index,item in enumerate(meta) if item['pkg_name'] + '.apk' == app]
        '''
        if len(apps_malware_index)>1:
            print(len(apps_malware_index))
        '''
        for i in apps_malware_index:
            if Y[i] == 1:
                selected_malware_app[app] = i
    path = os.path.join(config['features'] , 'sub_dataset/', 'accessible_malware_index.p')
    with open(path,'wb') as f:
        pickle.dump(selected_malware_app,f)
        
    
    selected_malware_index_accessible = [val for val in selected_malware_app.values()]
    selected_malware_index_inaccessible = [item for item in malware_index if item not in selected_malware_index_accessible]
    
    selected_malware_app_inaccessible = dict()
    for index,item in enumerate(malware_index):
        if item in selected_malware_index_inaccessible:
            selected_malware_app_inaccessible[apps_malware_total[index]] = item
    path = os.path.join(config['features'] , 'sub_dataset/', 'inaccessible_malware_index.p')
    with open(path, 'wb') as f:
        pickle.dump(selected_malware_app_inaccessible,f) 
    
    print("Start moving smaples to inaccessible/malware directory")
    apps_temp = selected_malware_app_inaccessible.keys()
    cp = 0
    for apk in apps_temp:
        app_path_src = os.path.join('C:/AndroidDatasets/AndoZoo',apk)
        app_path_des = os.path.join(config['apks'],'inaccessible/malware',apk)
        if os.path.exists(app_path_src) == True:
            shutil.copyfile(app_path_src, app_path_des)  
            os.remove(app_path_src)
            cp += 1
            print("cp - inaccessible/malware: " + str(cp))
    print("Finish moving smaples to inaccessible/malware directory")
    
        
    accessible_apps_index = selected_goodware_index_accessible + selected_malware_index_accessible
    X_accessible = [x for idx,x in enumerate(X) if idx in accessible_apps_index]
    path = os.path.join(config['features_accessible'],'accessible-dataset-X.json')
    with open(path, 'w') as f:
        json.dump(X_accessible,f)
    
    Y_accessible = [y for idx,y in enumerate(Y) if idx in accessible_apps_index]
    path = os.path.join(config['features_accessible'],'accessible-dataset-Y.json')
    with open(path, 'w') as f:
        json.dump(Y_accessible,f)
    
    meta_accessible = [m for idx,m in enumerate(meta) if idx in accessible_apps_index]
    path = os.path.join(config['features_accessible'],'accessible-dataset-meta.json')
    with open(path, 'w') as f:
        json.dump(meta_accessible,f)
        
    
    inaccessible_apps_index = selected_goodware_index_inaccessible + selected_malware_index_inaccessible
    X_inaccessible = [x for idx,x in enumerate(X) if idx in inaccessible_apps_index]
    path = os.path.join(config['features_inaccessible'],'inaccessible-dataset-X.json')
    with open(path, 'w') as f:
        json.dump(X_inaccessible,f)
    
    Y_inaccessible = [y for idx,y in enumerate(Y) if idx in inaccessible_apps_index]
    path = os.path.join(config['features_inaccessible'],'inaccessible-dataset-Y.json')
    with open(path, 'w') as f:
        json.dump(Y_inaccessible,f)
    
    meta_inaccessible = [m for idx,m in enumerate(meta) if idx in inaccessible_apps_index]
    path = os.path.join(config['features_inaccessible'],'inaccessible-dataset-meta.json')
    with open(path, 'w') as f:
        json.dump(meta_inaccessible,f)


def extract_mamadriod_features():
    
    goodware_path = list()
    apps = os.listdir(os.path.join(config['apks'],'inaccessible/normal'))
    for app in apps:
        goodware_path.append(os.path.join(config['apks'],'inaccessible/normal',app))
    
    malware_path = list()
    apps = os.listdir(os.path.join(config['apks'],'inaccessible/malware'))
    for app in apps:
        malware_path.append(os.path.join(config['apks'],'inaccessible/malware',app))
    
    apks_path_for_mamadroid = malware_path + goodware_path     
    random.shuffle(apks_path_for_mamadroid)
    db = "dataset_inaccessible"
    
    path = os.path.join(config['apks'],'inaccessible','apks_path_for_mamadroid.p')    
    with open(path, 'wb') as f:
        pickle.dump(apks_path_for_mamadroid, f) 
        
    mamadroid.api_sequence_extraction(apks_path_for_mamadroid,db)
    dbs = list()
    dbs.append(db)
    MaMaStat.feature_extraction_markov_chain(dbs)
    
def extract_mamadriod_malware_features():   
   
    malware_path = list()
    apps = os.listdir(os.path.join(config['apks'],'accessible/malware'))
    for app in apps:
        malware_path.append(os.path.join(config['apks'],'accessible/malware',app))
    
    apks_path_for_mamadroid = malware_path     
    random.shuffle(apks_path_for_mamadroid)
    db = "dataset_accessible_malware"
    
    path = os.path.join(config['apks'],'accessible','apks_path_for_mamadroid.p')    
    with open(path, 'wb') as f:
        pickle.dump(apks_path_for_mamadroid, f) 
        
    #mamadroid.api_sequence_extraction(apks_path_for_mamadroid,db)
    dbs = list()
    dbs.append(db)
    MaMaStat.feature_extraction_markov_chain(dbs)

def extract_n_gram_features():
    apks = dict()
    goodware_path = list()
    apps = os.listdir(os.path.join(config['apks'],'accessible/normal'))
    for app in apps:
        goodware_path.append(os.path.join(config['apks'],'accessible/normal',app))
        apks[app] = 0
    
    malware_path = list()
    apps = os.listdir(os.path.join(config['apks'],'accessible/malware'))
    for app in apps:
        malware_path.append(os.path.join(config['apks'],'accessible/malware',app))
        apks[app] = 1
    
    frompath = malware_path + goodware_path      
    print("No apps: ",len(frompath))
    topath = config['tmp_dir'] + "smalis"
    batch_disasseble.disassemble(frompath, topath, 3000)     
    bytecode_extract.collect(topath,0)     
    print("start n-gram")
    n_gram.extract_n_gram(5, apks)    
    print("end n-gram")  
