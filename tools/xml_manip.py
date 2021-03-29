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

