import os
import time
import random
import shutil
import tempfile
import subprocess

import re
import numpy as np
import torch

from core.droidfeature import Apk2graphs
from core.droidfeature import sequence_generator as seq_gen
from tools import dex_manip, xml_manip, utils
from config import config, logging, ErrorHandler

random.seed(0)

logger = logging.getLogger('core.droidfeature.inverse_feature_extraction')
logger.addHandler(ErrorHandler)

TMP_DIR = '/tmp'

OP_INSERTION = '+'
OP_REMOVAL = '-'

REFLECTION_TEMPLATE = '''.class public Landroid/content/res/MethodReflection;
.super Ljava/lang/Object;
.source "MethodReflection.java"


# direct methods
.method public constructor <init>()V
    .locals 1

    .prologue
    .line 3
    invoke-direct {p0}, Ljava/lang/Object;-><init>()V

    return-void
.end method
'''
DEFAULT_SMALI_DIR = 'android/content/res/'  # the path corresponds to the reflection class set above

INSERTION_TEMPLATE = '''.method public {newMethodName}()V  
    .locals 1
    
    .prologue
    const/4 v0, 0x0
    
    .local v0, "{varRandName}":{apiClassName}
    :try_start_0
    {invokeType}  {{v0}}, {apiClassName}->{methodName}(){returnType}
    :try_end_0
    .catch Ljava/lang/Exception; {{:try_start_0 .. :try_end_0}} :catch_0
    
    :goto_0
    return-void
    
    :catch_0
    move-exception v0

    goto :goto_0
.end method
'''

MANIFEST = "AndroidManifest.xml"
ENTRY_METHOD_STATEMENT = 'public onBind(Landroid/content/Intent;)Landroid/os/IBinder;'
EMPTY_SERVICE_BODY = '''.class public L{fullClassName}
.super Landroid/app/Service;
.source "{className}.java"

# direct methods
.method public constructor <init>()V
    .locals 0

    .line 8
    invoke-direct {p0}, Landroid/app/Service;-><init>()V

    .line 9
    return-void
.end method


.method {entryMethodStatement}
    .locals 2
    .param p1, "intent"    # Landroid/content/Intent;

    .line 14
    new-instance v0, Ljava/lang/UnsupportedOperationException;

    const-string v1, "Not yet implemented"

    invoke-direct {{v0, v1}}, Ljava/lang/UnsupportedOperationException;-><init>(Ljava/lang/String;)V

    throw v0
.end method

'''


class InverseDroidFeature(object):
    vocab, vocab_info = None, None

    def __init__(self):
        meta_data_saving_dir = config.get('dataset', 'intermediate')
        naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
        feature_extractor = Apk2graphs(naive_data_saving_dir, meta_data_saving_dir)
        InverseDroidFeature.vocab, InverseDroidFeature.vocab_info = feature_extractor.get_vocab()
        self.vocab = InverseDroidFeature.vocab
        self.vocab_info = InverseDroidFeature.vocab_info

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

    def inverse_map_manipulation(self, x_mod):
        """
        map the numerical manipulation to operation tuples (i.e., (api name, '+') or (api name, '-'))

        Parameters
        --------
        @param x_mod, numerical manipulations (i.e., perturbations) on node features x of a sample
        """
        assert isinstance(x_mod, (torch.Tensor, np.ndarray))
        if isinstance(x_mod, torch.Tensor) and (not x_mod.is_sparse):
            x_mod = x_mod.detach().cpu().numpy()

        if isinstance(x_mod, np.ndarray):
            indices = np.nonzero(x_mod)
            values = x_mod[indices]
        else:
            indices = x_mod._indices()
            values = x_mod._values()

        num_cg = x_mod.shape[0]
        instruction = []
        for i in range(num_cg):
            vocab_ind = indices[1][indices[0] == i]
            api = map(self.vocab.__getitem__, vocab_ind)
            manip_x = values[indices[0] == i]
            op_info = map(lambda v: OP_INSERTION if v > 0 else OP_REMOVAL, manip_x)
            instruction.append(tuple(zip(api, op_info)))
        return instruction

    def modify(self, x_mod_instr, feature_path, app_path, save_dir=None):
        """
        model a sample

        Parameters
        --------
        @param x_mod_instr, a list of manipulations
        @param feature_path, String, feature file path
        @param app_path, String, app path
        @param save_dir, String, saving directory
        """
        cg_dict = seq_gen.read_from_disk(feature_path)
        # we use feature padding, leading to more items of instructions than call graph
        assert len(x_mod_instr) >= len(
            cg_dict), 'Expect items of instructions greater than call graph, but got {} vs {}'.format(
            len(x_mod_instr),
            len(cg_dict)
        )
        assert os.path.isfile(app_path)
        if save_dir is None:
            save_dir = os.path.join(TMP_DIR, 'adv_mal_cache')
        if not os.path.exists(save_dir):
            utils.mkdir(save_dir)
        with tempfile.TemporaryDirectory() as tmpdirname:
            dst_file = os.path.join(tmpdirname, os.path.splitext(os.path.basename(app_path))[0])
            cmd_response = subprocess.call("apktool -q d " + app_path + " -o " + dst_file, shell=True)
            if cmd_response != 0:
                logger.error("Unable to disassemble app {}".format(app_path))
                return
            for instruction, (root_call, cg) in zip(x_mod_instr, cg_dict.items()):
                for api_name, op in instruction:
                    if op == OP_REMOVAL:
                        self.remove_api(api_name, cg, dst_file)
                    else:
                        # A large scale of insertion will trigger unexpected issues such as method limitation in a class
                        self.insert_api(api_name, root_call, dst_file)
            dst_file_apk = os.path.join(save_dir, os.path.splitext(os.path.basename(app_path))[0] + '_adv')
            cmd_response = subprocess.call("apktool -q b " + dst_file + " -o " + dst_file_apk, shell=True)
            if cmd_response != 0:
                shutil.copytree(dst_file, os.path.join(TMP_DIR, os.path.basename(dst_file)),
                                dirs_exist_ok=True)
                logger.error("Unable to assemble app {} and move it to {}.".format(dst_file, TMP_DIR))
                return
            subprocess.call("jarsigner -sigalg MD5withRSA -digestalg SHA1 -keystore " + os.path.join(
                config.get("DEFAULT", 'project_root'), "core/droidfeature/res/resignKey.keystore") + \
                            " -storepass resignKey " + dst_file_apk + ' resignKey',
                            shell=True)

    @staticmethod
    def remove_api(api_name, call_graph, disassemble_dir):
        """
        remove an api

        Parameters
        --------
        @param api_name, composite of class name and method name
        @param call_graph, call graph
        @param disassemble_dir, the work directory
        """
        if not (api_name in call_graph.nodes()):
            logger.warning("No {} found in {}.".format(api_name, disassemble_dir))
            return
        api_tag = call_graph.nodes(data=True)[api_name]['tag'].pop()
        caller_class_name, caller_method_statement = seq_gen.get_caller_info(api_tag)
        smali_path_of_class = os.path.join(disassemble_dir + '/smali',
                                           caller_class_name.lstrip('L').rstrip(';') + '.smali')
        method_finder_flag = False
        fh = dex_manip.read_file_by_fileinput(smali_path_of_class, inplace=True)
        for line in fh:
            if line.strip() == caller_method_statement:
                method_finder_flag = True
                print(line.rstrip())
                continue
            if line.strip() == '.end method':
                method_finder_flag = False

            if method_finder_flag:
                invoke_match = re.search(
                    r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) {(?P<invokeParam>([vp0-9,. ]*?))}, (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                    line)
                if invoke_match is None:
                    print(line.rstrip())
                else:
                    api_name_invoked = invoke_match.group('invokeObject') + '->' + invoke_match.group('invokeMethod')
                    if api_name_invoked == api_name:
                        new_file_name = 'Ref' + dex_manip.random_name(seed=int(time.time()), code=api_name)
                        new_class_name = 'L' + DEFAULT_SMALI_DIR + new_file_name + ';'
                        ref_class_body = REFLECTION_TEMPLATE.replace('MethodReflection', new_file_name)
                        ref_class_body = dex_manip.change_invoke_by_ref(new_class_name,
                                                                        ref_class_body,  # append method
                                                                        invoke_match.group('invokeType'),
                                                                        invoke_match.group('invokeParam'),
                                                                        invoke_match.group('invokeObject'),
                                                                        invoke_match.group('invokeMethod'),
                                                                        invoke_match.group('invokeArgument'),
                                                                        invoke_match.group('invokeReturn')
                                                                        )
                        ref_smail_path = os.path.join(disassemble_dir + '/smali',
                                                      DEFAULT_SMALI_DIR + new_file_name + '.smali')
                        if not os.path.exists(os.path.dirname(ref_smail_path)):
                            utils.mkdir(os.path.dirname(ref_smail_path))
                        dex_manip.write_whole_file(ref_class_body, ref_smail_path)
                    else:
                        print(line.rstrip())
            else:
                print(line.rstrip())
        fh.close()

    @staticmethod
    def create_entry_point(disassemble_dir):
        """
        creat an empty service for injecting methods
        """
        service_name = dex_manip.random_name(int(time.time())) + dex_manip.random_name(int(time.time()) + 1)
        xml_tree = xml_manip.get_xmltree_by_ET(os.path.join(disassemble_dir, MANIFEST))
        msg, response, new_manifest_tree = xml_manip.insert_comp_manifest(xml_tree, 'service', service_name)
        if not response:
            logger.error("Unable to create a new entry point {}.".format(msg))
        else:
            # create the service class correspondingly
            package_name = xml_tree.getroot().get('package')
            full_classname = package_name.replace('.', '/') + service_name + ';'
            service_class_body = EMPTY_SERVICE_BODY.format(
                fullClassName=full_classname,
                className=service_name,
                entryMethodStatement=ENTRY_METHOD_STATEMENT
            )

            svc_class_path = os.path.join(disassemble_dir + '/smali',
                                          package_name.replace('.', '/') + '/' + service_name + '.smali')
            if not os.path.exists(svc_class_path):
                dex_manip.mkdir(os.path.dirname(svc_class_path))
            dex_manip.write_whole_file(service_class_body, svc_class_path)
            return 'L' + full_classname + '.method ' + ENTRY_METHOD_STATEMENT

    @staticmethod
    def insert_api(api_name, root_call, disassemble_dir):
        """
        insert an api.

        Parameters
        -------
        @param api_name, composite of class name and method name
        @param api_info, invoke information about api, obtaining by vocab_info
        @param root_call, a tuple of several root nodes (owing to the composite of several subgrashs)
        @param disassemble_dir, work directory
        """
        assert len(root_call) > 0, "Expect at least a root call."
        root_call = random.choice(root_call)
        root_class_name, caller_method_statement = root_call.split(';', 1)
        smali_path = os.path.join(disassemble_dir + '/smali',
                                  root_class_name.lstrip('L') + '.smali')
        assert os.path.exists(smali_path), 'Root call file {} is absent.'.format(smali_path)
        api_info = InverseDroidFeature.vocab_info[InverseDroidFeature.vocab.index(api_name)]
        class_name, method_name = api_name.split('->')
        invoke_types, return_classes = set(), set()
        for info in list(api_info):
            _match = re.search(
                r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                info)
            invoke_types.add(_match.group('invokeType'))
            return_classes.add(_match.group('invokeReturn'))

        if 'invoke-static' in invoke_types:
            invoke_type = 'invoke-static'
        elif 'invoke-static/range' in invoke_types:
            invoke_type = 'invoke-static/range'
        elif 'invoke-virtual/range' in invoke_types:
            invoke_type = 'invoke-virtual/range'
        else:
            invoke_type = 'invoke-virtual'
        return_class = return_classes.pop()

        random_str = dex_manip.random_name(seed=int(time.time()), code=api_name)
        # handle the initialization methods: <init>, <cinit>
        new_method_name = method_name.lstrip('<').rstrip('>') + random_str
        new_method_body = INSERTION_TEMPLATE.format(
            newMethodName=new_method_name,
            methodName=method_name,
            varRandName=random_str,
            invokeType=invoke_type,
            apiClassName=class_name,
            returnType=return_class
        )

        method_finder_flag = False
        fh = dex_manip.read_file_by_fileinput(smali_path, inplace=True)
        for line in fh:
            print(line.rstrip())
            if line.strip() == caller_method_statement:
                method_finder_flag = True
                continue
            if method_finder_flag and line.strip() == '.end method':
                method_finder_flag = False
                # issue: injection ruins the correct line number in smali codes
                print('\n')
                print(new_method_body)
                continue

            invoke_virtual = 'invoke-virtual'
            if method_finder_flag and '.locals' in line:
                reg_match = re.match(r'^[ ]*?(.locals)[ ]*?(?P<regNumber>\d)', line)
                if int(reg_match.group('regNumber')) > 15:
                    invoke_virtual = 'invoke-virtual/range'

            if method_finder_flag:
                if re.match(r'^[ ]*?(.prologue)', line) is not None:
                    print('    ' + invoke_virtual + ' {p0}, ' + root_class_name + ';->' + new_method_name + '()V' + '\n')
        fh.close()


if __name__ == '__main__':
    # w = 'Landroid/content/Context;->startService(Landroid/content/Intent;)Landroid/content/ComponentName;'
    # w_info = set({'invoke-virtual Landroid/content/Context;->startService(Landroid/content/Intent;)Landroid/content/ComponentName;','invoke-virtual/range Landroid/content/Context;->startService(Landroid/content/Intent;)Landroid/content/ComponentName;'})
    # w = 'Ljava/lang/Class;->getDeclaredMethod'
    # w_info = set({'invoke-virtual Ljava/lang/Class;->getDeclaredMethod(Ljava/lang/String; [Ljava/lang/Class;)Ljava/lang/reflect/Method;',
    # 'invoke-virtual/range Ljava/lang/Class;->getDeclaredMethod(Ljava/lang/String; [Ljava/lang/Class;)Ljava/lang/reflect/Method;'})
    # InverseDroidFeature.approx_check_public_method(w, w_info)
    inverse_droid = InverseDroidFeature()
    print(inverse_droid.get_interdependent_apis())
