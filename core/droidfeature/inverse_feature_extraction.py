import re
import numpy as np

from core.droidfeature import Apk2graphs
from core.droidfeature import sequence_generator as seq_gen
from config import config


class InverseDroidFeature(object):
    def __init__(self):
        meta_data_saving_dir = config.get('drebin', 'intermediate')
        naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
        feature_extractor = Apk2graphs(naive_data_saving_dir, meta_data_saving_dir)
        self.vocab, self.vocab_info = feature_extractor.get_vocab()

    def get_manipulation(self):
        """
        We consider all apis are insertable and the apis that have public descriptor can be hidden by java reflection.
        For efficiency and simplicity consideration, this function only returns a mask to filter out the apis that are non-refelectable.
        This means the value "1" in the mask vector corresponds to a reflectable api, and "0" means otherwise.
        """
        manipulation = np.zeros((len(self.vocab),), dtype=np.float32)
        for i, v, v_info in zip(range(len(self.vocab)), self.vocab, self.vocab_info):
            if self.approx_check_public_method(v, v_info):
                manipulation[i] = 1.
        return manipulation

    def get_interdependent_apis(self):
        """
        For api insertion, no interdependent apis are considered. For api removal, getClass, getMethod and Invoke methods are used
        """
        interdependent_apis = ['Ljava/lang/Object;->getClass', 'Ljava/lang/Class;->getMethod',
                               'Ljava/lang/reflect/Method;->invoke']
        omega = [self.vocab.index(api) for api in interdependent_apis]
        return omega

    @staticmethod
    def merge_features(cg_dict1, cg_dict2):
        # avoid duplication of root call
        for root_call, _ in cg_dict1.items():
            if root_call in cg_dict2.keys():
                cg_dict2.pop(root_call)
        return {**cg_dict1, **cg_dict2}

    @staticmethod
    def approx_check_public_method(word, word_info):
        assert isinstance(word, str) and isinstance(word_info, set)
        # see: https://docs.oracle.com/javase/specs/jvms/se10/html/jvms-2.html#jvms-2.12
        if re.search(r'\<init\>|\<clinit\>', word) is None and \
                re.search(r'Ljava\/lang\/reflect\/|Ljava\/lang\/Class\;|Ljava\/lang\/Object\;', word) is None and any(
            [re.search(r'invoke\-virtual|invoke\-static|invoke\-interface', info) for info in word_info]):
            return True

    def generate_mod_instruction(self, sample_paths, perturbations):
        '''
        generate the instructions for samples in the attack list
        :param sample_paths: the list of file path
        :param perturbations: numerical perturbations on the un-normalized feature space, type: np.ndarray
        :return: {sample_path1: [meta_instruction1, ...], sample_path2: [meta_instruction1, ...],...}
        '''
        assert len(sample_paths) == len(perturbations)

        pass


if __name__ == '__main__':
    # w = 'Landroid/content/Context;->startService(Landroid/content/Intent;)Landroid/content/ComponentName;'
    # w_info = set({'invoke-virtual Landroid/content/Context;->startService(Landroid/content/Intent;)Landroid/content/ComponentName;','invoke-virtual/range Landroid/content/Context;->startService(Landroid/content/Intent;)Landroid/content/ComponentName;'})
    # w = 'Ljava/lang/Class;->getDeclaredMethod'
    # w_info = set({'invoke-virtual Ljava/lang/Class;->getDeclaredMethod(Ljava/lang/String; [Ljava/lang/Class;)Ljava/lang/reflect/Method;',
    # 'invoke-virtual/range Ljava/lang/Class;->getDeclaredMethod(Ljava/lang/String; [Ljava/lang/Class;)Ljava/lang/reflect/Method;'})
    # InverseDroidFeature.approx_check_public_method(w, w_info)
    inverse_droid = InverseDroidFeature()
    print(inverse_droid.get_interdependent_apis())
