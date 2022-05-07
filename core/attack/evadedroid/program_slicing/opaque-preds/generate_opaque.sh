#!/bin/bash

path_to_sootclasses_trunk_jar_with_dependencies_jar="sootclasses-trunk-jar-with-dependencies.jar"
classpath_with_java_8_openjdk_amd64_jre_lib_rt_jar=".:/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/rt.jar"

javac Opaque.java
java -cp $path_to_sootclasses_trunk_jar_with_dependencies_jar soot.Main -cp $classpath_with_java_8_openjdk_amd64_jre_lib_rt_jar -f jimple Opaque
