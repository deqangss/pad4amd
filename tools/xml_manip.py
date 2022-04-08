"""
feasible manipulations on androidmanifest.xml files: codes are from: https://github.com/deqangss/adv-dnn-ens-malware
"""

import os
import sys

import xml.etree.ElementTree as ET
from tools.utils import *

NAMESPACE = '{http://schemas.android.com/apk/res/android}'
ET.register_namespace('android', "http://schemas.android.com/apk/res/android")


def get_xmltree_by_ET(xml_path):
    """
    read the manifest.xml
    :param disassembly_root: the root folder
    :return: ET element
    """
    try:
        if os.path.isfile(xml_path):
            with open(xml_path, 'rb') as fr:
                parser = ET.XMLParser(encoding="utf-8")
                return ET.parse(fr, parser=parser)
        else:
            raise e.FileNotFound("Error: No such file '{}'.".format(xml_path))
    except IOError:
        raise IOError("Unable to load xml file from {}".format(xml_path))


def insert_comp_manifest(manifest_ET_tree, comp_type, comp_spec_name, mod_count=1):
    """
    insert a component into manifest.xml
    :param manifest_ET_tree: manifest ElementTree element
    :param comp_type: component types
    :param comp_spec_name: component name
    :return: info, True or False, ET tree of manifest
    """
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application == None:
        application = ET.SubElement(root, "application")
    comp_elems = application.findall(comp_type)

    comp_names = [elem.get(NAMESPACE + "name") for elem in comp_elems]

    if comp_spec_name in comp_names:
        MSG = 'Repetition allowed:{}/\'{}\'.'.format(comp_type, comp_spec_name)
        for t in range(mod_count):
            ET.SubElement(application, comp_type).set(NAMESPACE + "name", comp_spec_name)
        return MSG, False, manifest_ET_tree
    for t in range(mod_count):
        ET.SubElement(application, comp_type).set(NAMESPACE + "name", comp_spec_name)
    MSG = "Component inserted Successfully."
    return MSG, True, manifest_ET_tree


def insert_intent_manifest(manifest_ET_tree, comp_type, intent_spec_name, mod_count=1):
    """
    insert an intent-filter into a component of manifest.xml
    :param manifest_ET_tree: manifest ElementTree element
    :param comp_type: component types
    :param intent_spec_name: intent-filter action name
    :return: info, True or False, ET tree of manifest
    """
    root = manifest_ET_tree.getroot()
    application = root.find("application")
    if application == None:
        application = ET.SubElement(root, "application")
    comp_elems = application.findall(comp_type)

    comp_names = [elem.get(NAMESPACE + "name") for elem in comp_elems]

    comp_spec_name = 'abc'
    count = 0
    while count <= 1000:
        rdm_seed = random.randint(1, 23456)
        random.seed(rdm_seed)
        comp_spec_name = random_string(intent_spec_name) + random_name(random.randint(1, 23456)) + random_name(
            random.randint(1, 23456))
        count = count + 1
        if comp_spec_name not in comp_names:
            break

    comp_tree = ET.SubElement(application, comp_type)
    comp_tree.set(NAMESPACE + "name", comp_spec_name)

    for t in range(mod_count):
        intent_tree = ET.SubElement(comp_tree, 'intent-filter')
        ET.SubElement(intent_tree, "action").set(NAMESPACE + "name", intent_spec_name)
    return manifest_ET_tree


def insert_elem_manifest(manifest_ET_tree, elem_type, elem_spec_name, mod_count=1):
    root = manifest_ET_tree.getroot()
    all_elems = root.findall(elem_type)

    elem_names = [elem.get(NAMESPACE + "name") for elem in all_elems]

    if elem_spec_name in elem_names:
        for t in range(mod_count):
            ET.SubElement(root, elem_type).set(NAMESPACE + "name", elem_spec_name)

    for t in range(mod_count):
        ET.SubElement(root, elem_type).set(NAMESPACE + "name", elem_spec_name)
    return manifest_ET_tree


def dump_xml(save_path, et_tree):
    try:
        if os.path.isfile(save_path):
            with open(save_path, "wb") as fw:
                et_tree.write(fw, encoding="UTF-8", xml_declaration=True)
    except Exception as ex:
        raise IOError("Unable to dump xml file {}:{}.".format(save_path, str(ex)))