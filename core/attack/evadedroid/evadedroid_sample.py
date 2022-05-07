# -*- coding: utf-8 -*-
"""
The data structure of EvadeDroid's output 
"""


class APK:
    def __init__(self,app_name,malware_label, adv_malware_label, 
                 number_of_queries, percentage_increasing_size, 
                 number_of_features_malware,number_of_features_adv_malware,
                 number_of_features_adv_malware_per_query,
                 number_of_api_calls_malware,number_of_api_calls_adv_malware,
                 number_of_api_calls_adv_malware_per_query, transformations,
                 intact_due_to_soot_error,execution_time, classified_with_hard_label,query_time):
        self.app_name = app_name
        self.malware_label = malware_label
        self.adv_malware_label = adv_malware_label
        self.number_of_queries = number_of_queries
        self.percentage_increasing_size = percentage_increasing_size
        self.number_of_features_malware = number_of_features_malware
        self.number_of_features_adv_malware = number_of_features_adv_malware
        self.number_of_features_adv_malware_per_query = number_of_features_adv_malware_per_query
        self.number_of_api_calls_malware = number_of_api_calls_malware
        self.number_of_api_calls_adv_malware = number_of_api_calls_adv_malware
        self.number_of_api_calls_adv_malware_per_query = number_of_api_calls_adv_malware_per_query
        self.transformations = transformations
        self.intact_due_to_soot_error = intact_due_to_soot_error
        self.execution_time =  execution_time
        self.classified_with_hard_label = classified_with_hard_label
        self.query_time = query_time
        
    def compute_number_of_modified_feature_per_query(self):
        modified_features = list()
        last_no_features = self.number_of_features_malware
        for i,item in enumerate(self.number_of_features_adv_malware_per_query):
            diff_features = item - last_no_features
            modified_features.append(diff_features)
            last_no_features = item
        return modified_features
    
    def compute_number_of_modified_api_calls_per_query(self):
        modified_api_calls = list()
        last_no_api_calls = self.number_of_api_calls_adv_malware_per_query
        for i,item in enumerate(self.number_of_api_calls_adv_malware_per_query):
            diff_api_calls = item - last_no_api_calls
            modified_api_calls.append(diff_api_calls)
            last_no_api_calls = item
        return modified_api_calls
        