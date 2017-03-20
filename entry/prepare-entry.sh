#!/bin/bash
#
# file: prepare-entry.sh

set -e
set -o pipefail

cp ../LICENSE LICENSE.txt

echo "==== running entry script on validation set ===="
validation=/deep/group/med/alivecor/sample2017/validation

rm -f answers.txt
for r in `cat $validation/RECORDS`; do
    echo $r
    ln -sf $validation/$r.hea .
    ln -sf $validation/$r.mat .
    ./next.sh $r
    rm $r.hea $r.mat
done
