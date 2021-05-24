"""
generate API call sequences for an APK
"""

from os import path, getcwd
import warnings

import collections
from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import Analysis, ExternalMethod
from androguard.core.bytecodes.apk import APK
import networkx as nx
import time

import re

from tools.utils import dump_pickle, read_pickle, java_class_name2smali_name, \
    read_txt, retrive_files_set, remove_duplicate

ANDROID_LIFE_CIRCLE_METHODS = ['onCreate',
                               'onStart',
                               'onResume',
                               'onRestart',
                               'onPause',
                               'onStop',
                               'onDestroy'
                               ]

API_SIMLI_TAGS = ['Landroid/',
                  'Lcom/google/android/',
                  'Ldalvik/',
                  'Lorg/apache/',
                  'Lorg/json/',
                  'Lorg/w3c/dom/',
                  'Lorg/xml/sax',
                  'Lorg/xmlpull/v1/',
                  'Ljunit/',
                  'Ljava/net',
                  'Ljava/io/IOException',
                  'Ljava/lang/Runtime',
                  'Ljava/io/FileOutputStream',
                  'Ljava/lang/Runtime',
                  'Ljava/lang/System',
                  'Ljavax/crypto',
                  'Ljava/lang/reflect/',
                  'getClass',
                  'getConstructor',
                  'getField',
                  'getMethod',
                  'getDeclaredMethod',
                  'getDeclaredField',
                  ]

dir_path = path.dirname(path.realpath(__file__))
path_to_lib_type_1 = path.join(dir_path + '/res/liblist_threshold_10.txt')
Third_part_libraries_ALL = ['L' + lib_cnt.split(';')[0].strip('"').lstrip('/') for lib_cnt in read_txt(
    path_to_lib_type_1, mode='r')]

paths_to_lib_type2 = retrive_files_set(dir_path + '/res/libraries', '', 'txt')
for p in paths_to_lib_type2:
    Third_part_libraries_ALL.extend([java_class_name2smali_name(lib) for lib in read_txt(p, 'r')])

Third_part_libraries = []
for api in API_SIMLI_TAGS:
    for lib in Third_part_libraries_ALL:
        if lib.startswith(api):
            Third_part_libraries.append(lib)
Third_part_libraries = list(set(Third_part_libraries))
del Third_part_libraries_ALL

TAG_SPLITTER = '#.tag#'


def apk2graphs_wrapper(kwargs):
    try:
        return apk2graphs(*kwargs)
    except Exception as e:
        return e


def apk2graphs(apk_path, max_number_of_sequences=15000, max_recursive_depth=50, timeout=6, N=1, saving_path=None):
    """
    extract the api graph.
    Each API is represented by a string that is constructed by
    'invoke-type + ' ' + class_name + '->' + method_name + arguments + return_type+'#.tag.#'+ info of its entry point'

    :param apk_path: string, a path directs to an apk file, and otherwise an error is raised
    :param max_number_of_sequences: integer, the maximum number of searched sequences
    :param max_recursive_depth: integer, the maximum depth of visited methods
    :param timeout: integer, the elapsed time in minutes
    :param N: integer, the minimum number of graphs for an app if merging graphs,
    :param saving_path: string, a path directs to saving path
    """
    if not isinstance(apk_path, str):
        raise ValueError("Expected a path, but got {}".format(type(apk_path)))
    if not path.exists(apk_path):
        raise FileNotFoundError("Cannot find an apk file by following the path {}.".format(apk_path))
    if saving_path is None:
        warnings.warn("Save the features in current direction:{}".format(getcwd()))
        saving_path = path.join(getcwd(), 'api-graph')

    try:
        apk_path = path.abspath(apk_path)
        a, d, dx = AnalyzeAPK(apk_path)
    except Exception as e:
        raise ValueError("Fail to read and analyze the apk {}:{} ".format(apk_path, str(e)))

    # get entry points
    # 1. components as entry point
    entry_points_comp = get_comp_entry_points(a, d)

    # 2. build function caller-callee graph, this is used for find another kind of entry points
    entry_points_no_callers = get_no_caller_entry_points(dx)
    entry_points = entry_points_comp.copy()
    entry_points.extend([p for p in entry_points_no_callers if p not in entry_points_comp])

    # 3. if no entry points, randomly choose ones
    if len(entry_points) <= 0:
        entry_points = get_random_entry_points(dx)

    # 4. get system API graphs that are artificially made from API call sequences
    api_sequence_dict = get_api_call_graphs(entry_points,
                                            dx,
                                            max_number_of_sequences,
                                            max_recursive_depth,
                                            timeout
                                            )
    # for c,g in api_sequence_dict.items():
    #     for node in g.nodes(data=True):
    #         if node[0] == 'Ljava/io/IOException;->getMessage':
    #             print('tag: ', node[1])
    #             print('root: ', c)
    #             print('------------')

    # 5. merge graphs
    api_sequence_dict = merge_graphs(api_sequence_dict, N)

    # 6. saving the results
    if len(api_sequence_dict) > 0:
        save_to_disk(api_sequence_dict, saving_path)
        return saving_path
    else:
        raise ValueError("No graph found: " + apk_path)


def get_no_caller_entry_points(dx):
    """
    get the methods that have no callers, i.e., no methods call these methods
    :param dx: androidguard analysis object
    :return: a list of entry points
    :rtype: if the list is not empty, the type of entry point is 'androguard.core.analysis.analysis.MethodClassAnalysis'
    """
    if not isinstance(dx, Analysis):
        raise ValueError("Expected the object of androguard analysis.")

    mth_callers = collections.Counter()
    uninstantiated_classes = []
    for idx, c_obj in enumerate(dx.get_classes()):
        if c_obj.external:
            continue
        if len(c_obj.get_xref_from()) == 0:
            uninstantiated_classes.append(c_obj.orig_class.name)
        for m_obj in c_obj.get_methods():
            mth_callers[m_obj] += len(m_obj.get_xref_from())

    entry_points = []
    for c, k in mth_callers.items():
        """
        We do not decide the entry points exactly, and otherwise we over-approximate the entry points 
        e.g., static block are called by default, which are never found in the fashion of 'invoke-...'
        """
        if k <= 0:
            encoded_method = c.get_method()
            method_name = encoded_method.name
            class_name = encoded_method.class_name
            if method_name == '<init>':
                # assert class_name in uninstantiated_classes
                continue
            if method_name == '<clinit>' and class_name in uninstantiated_classes:
                continue
            if encoded_method not in entry_points:
                entry_points.append(encoded_method)
    return entry_points


def get_comp_entry_points(a, d):
    """
    get the components (e.g., activities, services, providers, and receivers) by searching androidmanifest.xml
    :param a: an instantiation of androguard.core.bytecodes.apk.APK
    :param d: an instantiation of androgurad.core.bytecodes.dvm.DalvikVMFormat
    :return: a list of entry points
    :rtype: if the list is not empty, the type of entry point is 'androguard.core.analysis.analysis.MethodClassAnalysis'
    """
    if not isinstance(a, APK):
        raise ValueError("Expected the object of Androidguard APK")
    xml_comp_name_list = a.get_activities() + a.get_services() + a.get_providers() + a.get_receivers()
    xml_comp_name_list = remove_duplicate(xml_comp_name_list)
    # map the java class name to the smali format
    pkg_name = a.get_package()
    smali_comp_name_list = []
    for n in xml_comp_name_list:
        if pkg_name in n:  # handle the case: pkg_name.comp_name
            smali_comp_name_list.append(java_class_name2smali_name(n))
        elif pkg_name not in n:  # handle the cases: .comp_name or comp_name
            if n.startswith('.'):
                smali_comp_name_list.append(pkg_name + n)
            else:
                smali_comp_name_list.append(pkg_name + '.' + n)
        else:
            raise ValueError("Cannot handle the component name {}.".format(n))

    # append the android life circle methods
    # search the encoded methods
    comp_entry_points = []
    for _d in d:
        for encoded_method in _d.get_methods():
            # note: here neglect arguments
            class_name = encoded_method.get_class_name()
            method_name = encoded_method.get_name()
            if (class_name in smali_comp_name_list) and (method_name in ANDROID_LIFE_CIRCLE_METHODS):
                # method_analysis = dx.get_method(encoded_method)
                # if method_analysis:
                comp_entry_points.append(encoded_method)

    return comp_entry_points


def get_random_entry_points(dx, number=100, seed=2345):
    """
    get the methods that have no callers, i.e., no methods call these methods
    :param dx: androidguard analysis object
    :param number: the maximum number of selected points
    :param seed: random seed
    :return: a list of entry points
    :rtype: if the list is not empty, the type of entry point is 'androguard.core.analysis.analysis.MethodClassAnalysis'
    """
    if not isinstance(dx, Analysis):
        raise ValueError("Expected the object of androguard analysis.")

    methods = []
    for idx, c_obj in enumerate(dx.get_classes()):
        for m_obj in c_obj.get_methods():
            methods.append(m_obj.get_method())
    if len(methods) <= 0:
        warnings.warn("No methods. Exit!")
        return []

    import random
    random.seed(seed)
    entry_points = list(set([random.choice(methods) for _ in range(number)]))
    return entry_points


def get_api_call_graphs(entry_points, dx, max_number_of_sequences, recursive_depth, timeout):
    """
    construct api call graphs using api sequences
    :param entry_points: list, a list of entry points, each of which is the type of 'androguard.core.analysis.analysis.MethodAnalysis'
    :param dx: an instantiation of 'androguard.core.analysis.analysis.Analysis'
    :param max_number_of_sequences: integer, the number of returned sequences
    :param recursive_depth: the maximum depth permitted by recursion function
    :param timeout: the elapsed time, in minutes
    :return: a list of api sequences
    """
    if not isinstance(entry_points, list):
        raise TypeError("Expect a list of entry points!")

    assert timeout > 0
    timeout = int(timeout)

    if len(entry_points) <= 0:
        entry_points = get_random_entry_points(dx)
        if len(entry_points) <= 0:
            warnings.warn("No entry points, return null.")
            return []

    def _api_of_interest(invoke_type, class_name, method_name, extra_description, method_tag, s):
        base_names = class_name.rstrip(';').split('/')
        base_str = base_names[0]
        base_name_list = [base_str]
        for n in base_names[1:]:
            base_str += '/' + n
            base_name_list.append(base_str)

        for base_name in base_name_list:
            if (base_name in Third_part_libraries) or \
                    (base_name + '/' in Third_part_libraries):
                return

        for api_tag in API_SIMLI_TAGS:
            if class_name.startswith(api_tag) or api_tag in method_name:
                s.append(
                    invoke_type + ' ' + class_name + '->' + method_name + extra_description + TAG_SPLITTER + method_tag)
                return

    def _get_method_tag(encoded_method):
        method_tag = encoded_method.class_name
        method_tag += '.method ' + encoded_method.access_flags_string + ' ' + encoded_method.name + encoded_method.proto
        return method_tag

    def _extend_graph(g, sequences):
        for s in sequences:
            if s is None or len(s) <= 0:
                continue
            api_name = get_api_name(s[0])
            if api_name not in g.nodes:
                g.add_node(api_name, tag={s[0]})
            else:
                g.nodes[api_name]['tag'].add(s[0])
            for idx in range(len(s) - 1):
                prev_api_name = get_api_name(s[idx])
                curr_api_name = get_api_name(s[idx + 1])
                if curr_api_name not in g.nodes:
                    g.add_node(curr_api_name, tag={s[idx + 1]})
                else:
                    g.nodes[curr_api_name]['tag'].add(s[idx + 1])
                g.add_edge(prev_api_name,
                           curr_api_name)  # make up the edge information using the sequential relationship

    sub_cg = nx.DiGraph()  # a sub-graph
    sub_api_sequences = []  # a sub-graph consists of multiple sub api sequences
    stack = []
    visited_non_leaf_nodes = []  # once a leaf node is visited, we do not visit it anymore
    visited_blocks = []  # once a block is visited, we do not visit it anymore
    start_time = time.time()

    # depth first search at the method level
    def _dfs(node, depth=0):
        # we only visit a api once in case of loop or function recursion
        if node in visited_non_leaf_nodes:
            return
        visited_non_leaf_nodes.append(node)
        # count depth
        depth += 1
        if depth >= recursive_depth:
            return
        if time.time() - start_time > int(60 * timeout):
            raise TimeoutError

        # depth first search at the block level and a method can have multiple blocks split by if, for, try...catch,...
        def _block_dfs(block, tag):
            if not block or block in visited_blocks:
                return
            if time.time() - start_time > int(60 * timeout):
                raise TimeoutError

            visited_blocks.append(block)
            sz_block_wise = len(stack)
            for instruction in block.get_instructions():
                smali_code = instruction.get_name() + ' { ' + instruction.get_output()  # on win32 platform, 'instruction.get_output()' triggers the memory exception 'exit code -1073741571 (0xC00000FD)' sometimes
                invoke_match = re.search(
                    r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) {(?P<invokeParam>([vp0-9,. ]*?)),? (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
                    smali_code)
                if invoke_match is None:
                    continue
                invoke_type = invoke_match.group('invokeType')
                class_name, method_name = invoke_match.group('invokeObject'), invoke_match.group('invokeMethod')
                arguments = '(' + invoke_match.group('invokeArgument') + ')' + invoke_match.group('invokeReturn')
                # get the encoded method accordingly
                encoded_method = dx.get_method_by_name(class_name, method_name, arguments)
                if encoded_method is None:
                    # todo: 'none' indicates the method could be implemented in the parent class
                    # the following lines over-estimate the call references
                    parent_class_name = dx.get_class_analysis(class_name).extends
                    _api_of_interest('invoke-super',
                                     parent_class_name,
                                     method_name,
                                     arguments,
                                     tag,
                                     stack)
                    continue
                if isinstance(encoded_method, ExternalMethod):
                    _api_of_interest(invoke_type,
                                     class_name,
                                     method_name,
                                     arguments,
                                     tag,
                                     stack)
                    continue
                _dfs(encoded_method, depth)
            # handle child blocks
            stack_size = len(stack)
            child_api_seqs = []
            for _1, _2, bc_child in block.childs:
                _block_dfs(bc_child, tag)
                changed_stack_size_child = len(stack) - stack_size
                if changed_stack_size_child > 0 and stack not in child_api_seqs:
                    child_api_seqs.append(stack.copy())
                for _ in range(changed_stack_size_child):
                    stack.pop()
            if block.get_exception_analysis():
                for exception in block.get_exception_analysis().exceptions:
                    _block_dfs(exception[-1], tag)
                    changed_stack_size_child = len(stack) - stack_size
                    if changed_stack_size_child > 0 and stack not in child_api_seqs:
                        child_api_seqs.append(stack.copy())
                    for _ in range(changed_stack_size_child):
                        stack.pop()
            if len(child_api_seqs) > 0:
                if len(sub_api_sequences) == 0:
                    sub_api_sequences.extend(child_api_seqs)
                    _extend_graph(sub_cg, sub_api_sequences)
                elif (len(sub_api_sequences) <= max_number_of_sequences) and (len(sub_api_sequences) > 0):
                    tmp_api_sequences = sub_api_sequences.copy()
                    sub_api_sequences.clear()
                    for tmp_api_seq in tmp_api_sequences:
                        for sub_seq in child_api_seqs:
                            sub_api_sequences.append(tmp_api_seq[-1:] + sub_seq)
                    del tmp_api_sequences
                    _extend_graph(sub_cg, sub_api_sequences)
                else:
                    pass
                # print(child_api_seqs)
                del child_api_seqs
                stack_size_changed = len(stack) - sz_block_wise
                for _ in range(stack_size_changed):
                    stack.pop()
            return

        analyzed_method = dx.get_method(node)  # node is encoded method
        method_tag = _get_method_tag(node)
        for basic_block in analyzed_method.basic_blocks.gets():
            _block_dfs(basic_block, method_tag)
        return

    cgs = collections.defaultdict(nx.DiGraph)  # a set of sub-graphs
    number_of_sequences = 0
    timeout_flag = False
    for root_call in entry_points:
        if isinstance(root_call, ExternalMethod):
            continue
        try:
            _dfs(root_call)
        except TimeoutError:
            warnings.warn("Timeout")
            timeout_flag = True
        finally:
            if (len(sub_api_sequences) <= 0) and (len(stack) > 0):
                _extend_graph(sub_cg, [stack])
            else:
                for api_seq in sub_api_sequences:
                    _extend_graph(sub_cg, [api_seq[-1:] + stack])
            number_of_sequences += len(sub_api_sequences)
            method_tag = _get_method_tag(root_call)
            cgs[method_tag] = sub_cg.copy()
            sub_api_sequences.clear()
            stack.clear()
            sub_cg.clear()

            if (number_of_sequences > max_number_of_sequences) and timeout_flag:
                return cgs
    return cgs


def merge_graphs(api_seq_dict, N=1):
    """
    merge graphs based on class names
    :param api_seq_dict: collections.defaultdict, keys: entry points, values: call graphs
    :param N: Integer, the minimum number of graphs for an app
    """

    root_nodes = list(api_seq_dict.keys())
    if len(root_nodes) <= N:
        return api_seq_dict

    if isinstance(root_nodes[0], tuple):
        class_names = [get_same_class_prefix(list(root_node)) for root_node in root_nodes]  # duplication exists
    elif isinstance(root_nodes[0], str):
        class_names = [root_node.split(';')[0].split('$')[0] for root_node in root_nodes]
    else:
        raise TypeError("Expect 'list' or 'tuple'.")

    def map_class_to_node(class_name):
        return [root_nodes[j] for j in \
                [i for i, e in enumerate(class_names) if e == class_name]]

    def merge(class_name_list):
        assert len(class_name_list) > 0

        new_root_node = ()
        g = api_seq_dict[map_class_to_node(class_name_list[0])[0]]
        for class_name in class_name_list:
            root_node_list = map_class_to_node(class_name)
            for rn in root_node_list:
                if not isinstance(rn, tuple):
                    new_root_node += (rn, )
                else:
                    new_root_node += rn
                g = nx.compose(g, api_seq_dict[rn])
        return new_root_node, g

    new_cg_dict = collections.defaultdict(nx.DiGraph)

    # merge all graph
    if N > 0 and N <= 1:
        rn, g = merge(sorted(set(class_names), key=class_names.index))  # remove duplicates and keep in order
        new_cg_dict[rn] = g
        return new_cg_dict

    # merge graphs based on class name
    max_class_length = max([class_name.count('/') for class_name in class_names])
    pre_class_name_list = []
    for idx in range(max_class_length):
        pre_class_name_set = set()
        pre_class_name_list.clear()
        for class_name in class_names:
            pre_class_name = class_name.rsplit('/', idx)[0]
            if pre_class_name not in pre_class_name_set:
                pre_class_name_list.append(pre_class_name)
            pre_class_name_set.add(pre_class_name)
        if len(pre_class_name_list) <= N:
            break
    if len(pre_class_name_list) <= N:
        for pre_class_name in pre_class_name_list:
            _class_names = [class_name for class_name in class_names if pre_class_name in class_name]
            rn, g = merge(sorted(set(_class_names), key=_class_names.index))
            new_cg_dict[rn] = g
    else:
        rn, g = merge(sorted(set(class_names), key=class_names.index))
        new_cg_dict[rn] = g
    return new_cg_dict


def save_to_disk(data, saving_path):
    dump_pickle(data, saving_path)


def read_from_disk(loading_path):
    return read_pickle(loading_path)


def get_api_name(node_tag):
    if not isinstance(node_tag, str):
        raise TypeError
    assert TAG_SPLITTER in node_tag
    api_info = node_tag.split(TAG_SPLITTER)[0]
    invoke_match = re.search(
        r'^([ ]*?)(?P<invokeType>invoke\-([^ ]*?)) (?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$',
        api_info)
    api_name = invoke_match.group('invokeObject') + '->' + invoke_match.group('invokeMethod')
    return api_name


def get_api_info(node_tag):
    if not isinstance(node_tag, str):
        raise TypeError
    assert TAG_SPLITTER in node_tag
    api_info = node_tag.split(TAG_SPLITTER)[0]
    return api_info


def get_caller_info(node_tag):
    if not isinstance(node_tag, str):
        raise TypeError
    assert TAG_SPLITTER in node_tag
    call_info = node_tag.split(TAG_SPLITTER)[1]
    class_name, method_statement = call_info.split(';', 1)
    # tailor tab issue that may be triggered by encodedmethod of androidguard
    method_statement = ';'.join([s.strip() for s in method_statement.split(';')])
    return class_name+';', method_statement


def get_api_tag(api_ivk_line, api_callee_class_name, api_callee_name):
    return api_ivk_line + TAG_SPLITTER + api_callee_class_name + api_callee_name


def get_same_class_prefix(entry_node_list):
    assert isinstance(entry_node_list, list)
    if len(entry_node_list) <= 0:
        return ''
    class_names = [a_node.split('.method')[0].rsplit('$')[0] for a_node in entry_node_list]
    a_class_name = class_names[0]
    n_names = a_class_name.count('/')
    for idx in range(n_names):
        pre_class_name = a_class_name.rsplit('/', idx)[0]
        if all([pre_class_name in class_name for class_name in class_names]):
            return pre_class_name
    else:
        return ''


def _main():
    rtn_str = apk2graphs(
        '/local_disk/data/Android/drebin/benign_samples/a8b37c627407d1444967828bcfe09d4a093dad48b5ebd964633baaf318de7916',
        200000,
        50,
        1,
        "./abc.cgs")
    print(rtn_str)


if __name__ == "__main__":
    import sys

    sys.exit(_main())
