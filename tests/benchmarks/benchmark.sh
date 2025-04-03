#!/bin/bash

PYPOREFLOW='PythonSlicer'

for i in 128 256 512 1024 2048 4096 8192 16384; do
    elapsed_time=$( ${PYPOREFLOW} -m py_pore_flow --nogui tests/resources/regular_networks/$i/ 2> /dev/null );
    time_part=${elapsed_time#*: };
    
    h=${time_part%%:*}
    m_s=${time_part#*:}
    m=${m_s%%:*}
    s=${m_s#*:}

    m=$((10#$m))
    s=$((10#$s))

    total_seconds=$(( h * 3600 + m * 60 + s ))

    echo $i $total_seconds
    echo $i $total_seconds >> benchmark.dat
done
