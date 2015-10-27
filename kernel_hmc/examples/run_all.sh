#!/bin/bash

# Runs all example python script in the current folder,
# prepending a matplotlib backend, such that plotting
# code in the example doesn't block the script, or fail
# the execution on servers without DISPLAY

# create temp file for all example scripts
LIST=$(mktemp /tmp/kernel_hmc_examples.XXXXXXXXXX) || { echo "Failed to create temp file"; exit 1; }

# find all example scripts
find . -type f -name 'demo_*.py' ! -name '__init__.py' > $LIST

# iterate over all scripts
while read name
do
	# prepend matplotlib backend that does not block
	echo "import matplotlib; matplotlib.use('Agg')\n" | cat - "$name" > "$name"_with_header

	# run
    echo Running example "$name"
    python "$name"_with_header > /dev/null
	
	# store return code
	retval=$?
	
	# clean up
	rm "$name"_with_header
	
	if [ $retval -ne 0 ]; then
		echo "Example $name failed with error code $retval."
		exit 1
	fi
done < $LIST

# clean up
rm $LIST
