import os
import time
import random
import shutil
import tempfile
import subprocess
import traceback
import string
import re
import numpy as np
import networkx as nx
import torch
from core.droidfeature import Apk2features, NULL_ID
from core.droidfeature import feature_gen as seq_gen
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

INSERTION_STATIC_TEMPLATE = '''.method public static {newMethodName}()V  
    .locals {numLocals:d}

    .prologue
    const/4 v0, 0x0
    
    .local v0, "a":I
    const/4 v1, 0x1
    if-ne v0, v1, :cond_0
    
    :try_start_0
{argInitialization}
    {invokeType} {{{paramRegisters}}}, {apiClassName}->{methodName}({argumentTypes}){returnType}
    :try_end_0
    .catch Ljava/lang/Exception; {{:try_start_0 .. :try_end_0}} :catch_0

{varEndCont}

    goto :goto_0

    :catch_0
    move-exception v0

    :cond_0
    :goto_0
    return-void
.end method
'''

INSERTION_TEMPLATE = '''.method public static {newMethodName}()V  
    .locals {numLocals:d}

    .prologue
    const/4 v0, 0x0
    
    .local v0, "a":I
    const/4 v1, 0x1
    if-ne v0, v1, :cond_0
    
    const/4 v0, 0x0
    
    .local v0, "{varRandName}":{apiClassName}
    :try_start_0
{argInitialization}
    {invokeType} {{{paramRegisters}}}, {apiClassName}->{methodName}({argumentTypes}){returnType}
    :try_end_0
    .catch Ljava/lang/Exception; {{:try_start_0 .. :try_end_0}} :catch_0

    .end local v0    # "{varRandName}":{apiClassName}
{varEndCont}

    goto :goto_0

    :catch_0
    move-exception v0
    
    :cond_0
    :goto_0
    return-void
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
    vocab, vocab_info, vocab_type = None, None, None

    def __init__(self, seed=0):
        random.seed(seed)
        meta_data_saving_dir = config.get('dataset', 'intermediate')
        naive_data_saving_dir = config.get('metadata', 'naive_data_pool')
        self.feature_extractor = Apk2features(naive_data_saving_dir, meta_data_saving_dir)
        InverseDroidFeature.vocab, InverseDroidFeature.vocab_info, InverseDroidFeature.vocab_type = \
            self.feature_extractor.get_vocab()
        self.vocab = InverseDroidFeature.vocab
        self.vocab_info = InverseDroidFeature.vocab_info
        self.vocab_type = InverseDroidFeature.vocab_type

    def get_manipulation(self):
        """
        All features are insertable and the apis that have public descriptor can be hidden by java reflection.
        For efficiency and simplicity consideration, this function only returns a mask to filter out the apis that are non-refelectable.
        This means the value "1" in the mask vector corresponds to a reflectable api, and "0" means otherwise.
        """
        manipulation = np.zeros((len(self.vocab),), dtype=np.float32)
        for i, v, v_info, v_type in zip(range(len(self.vocab)), self.vocab, self.vocab_info, self.vocab_type):
            if v_type in [seq_gen.ACTIVITY, seq_gen.SERVICE, seq_gen.RECEIVER, seq_gen.PROVIDER]:
                manipulation[i] = 1.
            if v_type == seq_gen.SYS_API and self.approx_check_public_method(v, v_info):
                manipulation[i] = 1.
        return manipulation

    def get_interdependent_apis(self):
        """
        For api insertion, no interdependent apis are considered. For api removal, getClass, getMethod and Invoke methods are used
        """
        interdependent_apis = ['Ljava/lang/Object;->getClass', 'Ljava/lang/Class;->getMethod',
                               'Ljava/lang/reflect/Method;->invoke']
        omega = [self.vocab.index(api) for api in interdependent_apis if api in self.vocab]
        return omega

    @staticmethod
    def merge_features(cg_dict1, cg_dict2):
        """
        randomly pick a graph from cg1 and inject graphs of cg_dict2 into it
        """
        n_src_cgs = len(cg_dict1)
        if n_src_cgs <= 0:
            return cg_dict2
        idx_mod = []
        for root_call, cg in cg_dict2.items():
            idx = random.choice(range(n_src_cgs))
            src_root_call, src_cg = list(cg_dict1.items())[idx]
            src_cg = nx.compose(src_cg, cg)
            cg_dict1[src_root_call] = src_cg
            idx_mod.append(idx)
        return cg_dict1, idx_mod

    @staticmethod
    def approx_check_public_method(word, word_info):
        assert isinstance(word, str) and isinstance(word_info, set)
        # see: https://docs.oracle.com/javase/specs/jvms/se10/html/jvms-2.html#jvms-2.12, we do not hide reflection-related API again
        if re.search(r'\<init\>|\<clinit\>', word) is None and \
                re.search(r'Ljava\/lang\/reflect\/|Ljava\/lang\/Object;->getClass|Ljava\/lang\/Class;->getMethod', word) is None and \
                all(
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
            apis = list(map(self.vocab.__getitem__, vocab_ind))
            if NULL_ID in apis:  # not an api
                apis = list(sorted(set(apis), key=apis.index))
                apis.remove(NULL_ID)
            manip_x = values[indices[0] == i]
            op_info = map(lambda v: OP_INSERTION if v > 0 else OP_REMOVAL, manip_x)
            instruction.append(tuple(zip(apis, op_info)))
        return instruction

    @staticmethod
    def modify_wrapper(args):
        try:
            return InverseDroidFeature.modify(*args)
        except Exception as e:
            traceback.print_stack()
            traceback.print_exc()
            return e

    @staticmethod
    def modify(x_mod_instr, feature_path, app_path, save_dir=None):
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
                        remove_api(api_name, cg, dst_file)
                    else:
                        # A large scale of insertion operations will trigger unexpected issues, such as method limitation in a class
                        insert_api(api_name, root_call, dst_file)
            dst_file_apk = os.path.join(save_dir, os.path.splitext(os.path.basename(app_path))[0] + '_adv')
            cmd_response = subprocess.call("apktool -q b " + dst_file + " -o " + dst_file_apk, shell=True)
            if cmd_response != 0:
                if os.path.exists(os.path.join(TMP_DIR, os.path.basename(dst_file))):
                    shutil.rmtree(os.path.join(TMP_DIR, os.path.basename(dst_file)))
                shutil.copytree(dst_file, os.path.join(TMP_DIR, os.path.basename(dst_file)))
                logger.error("Unable to assemble app {} and move it to {}.".format(dst_file, TMP_DIR))
                return False
            else:
                subprocess.call("jarsigner -sigalg MD5withRSA -digestalg SHA1 -keystore " + os.path.join(
                    config.get("DEFAULT", 'project_root'), "core/droidfeature/res/resignKey.keystore") + \
                                " -storepass resignKey " + dst_file_apk + ' resignKey',
                                shell=True)
                logger.info("Apk signed: {}.".format(dst_file_apk))
                return True


def remove_api(api_name, call_graph, disassemble_dir, coarse=True):
    """
    remove an api

    Parameters
    --------
    @param api_name, composite of class name and method name
    @param call_graph, call graph
    @param disassemble_dir, the work directory
    @param coarse, whether use reflection to all matched methods
    """
    if not (api_name in call_graph.nodes()):
        logger.warning("Removing {}, but got it non-found in {}.".format(api_name, disassemble_dir))
        return

    api_tag_set = call_graph.nodes(data=True)[api_name]['tag']
    # we attempt to obtain more relevant info about this api. Nonetheless, once there is class inheritance,
    # we cannot make it.
    if coarse:
        api_info_list = dex_manip.retrieve_api_caller_info(api_name, disassemble_dir)
        for api_info in api_info_list:
            api_tag_set.add(seq_gen.get_api_tag(api_info['ivk_method'],
                                                api_info['caller_cls_name'],
                                                api_info['caller_mth_stm']
                                                )
                            )
    api_class_set = set()
    for api_tag in api_tag_set:
        api_class_set.add(seq_gen.get_api_class(api_tag))

    for api_tag in api_tag_set:
        caller_class_name, caller_method_statement = seq_gen.get_caller_info(api_tag)
        smali_dirs = dex_manip.retrieve_smali_dirs(disassemble_dir)
        smali_path_of_class = None
        for smali_dir in smali_dirs:
            _path = os.path.join(smali_dir, caller_class_name.lstrip('L').rstrip(';') + '.smali')
            if os.path.exists(_path):
                smali_path_of_class = _path
                break
        if smali_path_of_class is None:
            logger.warning('File {} has no root call {}.'.format(disassemble_dir,
                                                                 caller_class_name.lstrip('L').rstrip(';') + '.smali'))
            continue
        method_finder_flag = False
        # note: owing to the inplace 'print', do not use std.out operation again until the following file is closed
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
                    invoked_mth_name = invoke_match.group('invokeMethod')
                    invoked_cls_name = invoke_match.group('invokeObject')
                    if (invoked_mth_name == api_name.split('->')[1]) and (invoked_cls_name in api_class_set):
                        cur_api_name = invoke_match.group('invokeObject') + '->' + invoke_match.group('invokeMethod')
                        new_file_name = 'Ref' + dex_manip.random_name(seed=int(time.time()), code=cur_api_name)
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

    api_info = InverseDroidFeature.vocab_info[InverseDroidFeature.vocab.index(api_name)]
    class_name, method_name = api_name.split('->')
    invoke_types, return_classes, arguments = list(), list(), list()
    for info in list(api_info):
        _match = re.search(
            r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
            info)
        invoke_types.append(_match.group('invokeType'))
        return_classes.append(_match.group('invokeReturn'))
        arguments.append(_match.group('invokeArgument'))

    api_idx = 0
    is_simplified_vars_register = False
    if 'invoke-virtual' in invoke_types:
        invoke_type = 'invoke-virtual'
        api_idx = invoke_types.index('invoke-virtual')
    elif 'invoke-virtual/range' in invoke_types:
        invoke_type = 'invoke-virtual/range'
        api_idx = invoke_types.index('invoke-virtual/range')
        is_simplified_vars_register = True
    elif 'invoke-interface' in invoke_types:
        invoke_type = 'invoke-interface'
        api_idx = invoke_types.index('invoke-interface')
    elif 'invoke-interface/range' in invoke_types:
        invoke_type = 'invoke-interface/range'
        api_idx = invoke_types.index('invoke-interface/range')
        is_simplified_vars_register = True
    elif 'invoke-static' in invoke_types:
        invoke_type = 'invoke-static'
    elif 'invoke-static/range' in invoke_types:
        invoke_type = 'invoke-static/range'
        is_simplified_vars_register = True
    # elif 'invoke-super' in invoke_types:
    #     invoke_type = 'invoke-super'
    #     api_idx = invoke_types.index('invoke-super')
    #     return
    # elif 'invoke-super/range' in invoke_types:
    #     invoke_type = 'invoke-super/range'
    #     api_idx = invoke_types.index('invoke-super/range')
    #     is_simplified_vars_register = True
    #     return
    # elif 'invoke-direct' in invoke_types:
    #     invoke_type = 'invoke-direct'
    #     api_idx = invoke_types.index('invoke-direct')
    # elif 'invoke-direct/range' in invoke_types:
    #     invoke_type = 'invoke-direct/range'
    #     api_idx = invoke_types.index('invoke-direct/range')
    #     is_simplified_vars_register = True
    else:
        # logger.warning('Neglect invocation type(s):{}'.format(' '.join(invoke_types)))
        return

    assert len(invoke_types) > 0, 'No api details.'
    return_class = return_classes[api_idx]
    argument = arguments[api_idx]
    # handle arguments
    # variable initialization
    arg_types = argument.split(' ')  # this is specific to androguard
    var_initial_content = ''
    if 'invoke-static' not in invoke_type:
        var_registers = 'v0'
    else:
        var_registers = ''
    var_count = 0
    var_end_content = ''
    for arg_type in arg_types:
        arg_type = arg_type.strip()
        if arg_type == '':
            continue
        if var_count >= 52:
            raise ValueError("Too much arguments of API method {}.".format(api_name))
        if arg_type[-1] == ';':
            var_count += 1
            var_value = dex_manip.smaliClassTInitialV.format(varNum=var_count)
            var_statement = dex_manip.VAR_STATEMENT_TEMPLATE.format(varNum=var_count,
                                                                    varName=string.ascii_letters[var_count - 1],
                                                                    varType=arg_type)
            var_end = dex_manip.VAR_END_TEMPLATE.format(varNum=var_count,
                                                        varName=string.ascii_letters[var_count - 1],
                                                        varType=arg_type
                                                        )
            var_registers += ', v{:d}'.format(var_count)
        elif arg_type in list(dex_manip.smaliBasicTInitialV.keys()):
            var_count += 1
            var_value = dex_manip.smaliBasicTInitialV[arg_type].format(varNum=var_count)
            var_statement = dex_manip.VAR_STATEMENT_TEMPLATE.format(varNum=var_count,
                                                                    varName=string.ascii_letters[var_count - 1],
                                                                    varType=arg_type)
            var_end = dex_manip.VAR_END_TEMPLATE.format(varNum=var_count,
                                                        varName=string.ascii_letters[var_count - 1],
                                                        varType=arg_type
                                                        )
            var_registers += ', v{:d}'.format(var_count)
            if dex_manip.is_wide_type(arg_type):  # 'const-wide'
                var_count += 1
                var_registers += ', v{:d}'.format(var_count)

        elif arg_type in list(dex_manip.smaliArrayInitialV.keys()):
            var_count += 1
            var_value = dex_manip.smaliArrayInitialV[arg_type].format(varNum=var_count)
            var_statement = dex_manip.VAR_STATEMENT_TEMPLATE.format(varNum=var_count,
                                                                    varName=string.ascii_letters[var_count - 1],
                                                                    varType=arg_type)
            var_end = dex_manip.VAR_END_TEMPLATE.format(varNum=var_count,
                                                        varName=string.ascii_letters[var_count - 1],
                                                        varType=arg_type
                                                        )
            var_registers += ', v{:d}'.format(var_count)

        var_initial_content += '\n' + var_value + '\n' + var_statement + '\n'
        var_end_content += var_end + '\n'
    var_registers = var_registers.lstrip(',').strip()
    arg_types_used = ''.join(arg_types[:var_count])

    random_str = dex_manip.random_name(seed=int(time.time()), code=api_name)
    # handle the initialization methods: <init>, <cinit>
    new_method_name = method_name.lstrip('<').rstrip('>') + random_str
    if invoke_type != 'invoke-static' and invoke_type != 'invoke-static/range':
        if var_count >= 5 and '/range' not in invoke_type:
            invoke_type += '/range'
            is_simplified_vars_register = True
        if is_simplified_vars_register:
            var_registers = 'v0 .. v{:d}'.format(var_count)
        var_count = var_count + 1 if var_count >= 1 else 2
        new_method_body = INSERTION_TEMPLATE.format(
            newMethodName=new_method_name,
            numLocals=var_count,
            argInitialization=var_initial_content,
            methodName=method_name,
            varRandName=random_str,
            invokeType=invoke_type,
            paramRegisters=var_registers,
            apiClassName=class_name,
            argumentTypes=arg_types_used,
            returnType=return_class,
            varEndCont=var_end_content
        )
    else:
        if var_count > 5 and '/range' not in invoke_type:
            invoke_type += '/range'
            is_simplified_vars_register = True
        if is_simplified_vars_register:
            var_registers = 'v1 .. v{:d}'.format(var_count)
        var_count = var_count + 1 if var_count >= 1 else 2
        new_method_body = INSERTION_STATIC_TEMPLATE.format(
            newMethodName=new_method_name,
            numLocals=var_count,
            argInitialization=var_initial_content,
            methodName=method_name,
            varRandName=random_str,
            invokeType=invoke_type,
            paramRegisters=var_registers,
            apiClassName=class_name,
            argumentTypes=arg_types_used,
            returnType=return_class,
            varEndCont=var_end_content
        )

    injection_done = False
    if isinstance(root_call, str):
        root_call = (root_call,)
    for rc in root_call:
        root_class_name, caller_method_statement = rc.split(';', 1)
        method_match = re.match(
            r'^([ ]*?)\.method\s+(?P<methodPre>([^ ].*?))\((?P<methodArg>(.*?))\)(?P<methodRtn>(.*?))$',
            caller_method_statement)
        caller_method_statement = '.method ' + method_match['methodPre'].strip() + '(' + method_match[
            'methodArg'].strip().replace(' ', '') + ')' + method_match['methodRtn'].strip()

        smali_dirs = dex_manip.retrieve_smali_dirs(disassemble_dir)
        smali_path = None
        for smali_dir in smali_dirs:
            _path = os.path.join(smali_dir, root_class_name.lstrip('L') + '.smali')
            if os.path.exists(_path):
                smali_path = _path
                break
        if smali_path is None:
            logger.warning('root call file {} is absent.'.format(smali_path))
            continue

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

            invoke_static = 'invoke-static'
            if method_finder_flag and '.locals' in line:
                reg_match = re.match(r'^[ ]*?(.locals)[ ]*?(?P<regNumber>\d+)', line)
                if reg_match is not None and int(reg_match.group('regNumber')) > 15:
                    invoke_static = 'invoke-static/range'

            if method_finder_flag:
                if re.match(r'^[ ]*?(.locals)', line) is not None:
                    print(
                        '    ' + invoke_static + ' {}, ' + root_class_name + ';->' + new_method_name + '()V' + '\n')
                    injection_done = True
        fh.close()
        if injection_done:
            break
