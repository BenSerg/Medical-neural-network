#!/bin/bash

read -p "Введите аргументы для функции run (разделите их пробелами): " args

python3 -c "from run import run; run('prepcache.LunaPrepCacheApp', *'$args'.split())"

