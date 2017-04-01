#!/bin/bash
#
# file: prepare-entry.sh

set -e
set -o pipefail

mkdir entry && cd entry

## Copy source code files
echo `pwd`
cp ../../LICENSE LICENSE.txt
cp ../setup.sh .
cp ../next.sh .
cp ../AUTHORS.txt .
cp ../dependencies.txt .
for f in 'evaler' 'loader' 'network'
do
    cp ../../${f}.py .
done

## Copy model files
model_dir=$1
for f in 'config.json'  'loader.pkl'  'best_model.epoch.data-*'  'best_model.epoch.index'
do
    cp $model_dir/$f .
done

echo "==== running entry script on validation set ===="
validation=/deep/group/med/alivecor/sample2017/validation

for r in `cat $validation/RECORDS`; do
    echo $r
    ln -sf $validation/$r.hea .
    ln -sf $validation/$r.mat .
    ./next.sh $r
    rm $r.hea $r.mat
done

## Make zip
rm -r __pycache__
zip -r entry.zip entry

## Cleanup
# rm -r entry
