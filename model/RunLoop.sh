#!/usr/bin/env bash
file=$1
shift
command=$@
echo "Running command: [$command] until file [$file] is created."
while [ ! -f "$file" ] 
do
    $command
    sleep 3
done
echo "file [$file] is created. Exiting successfully."
    
