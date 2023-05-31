#!/bin/bash

export XRD_NETWORKSTACK=IPv4
export X509_USER_PROXY=$HOME/tmp/x509up
cd MAINDIRECTORY

python3 run.py --processor PROCESSOR --executor EXECUTOR --workers WORKERS --channel CHANNEL --nfiles NFILES --year YEAR --redirector REDIRECTOR --output_location OUTPUTLOCATION --tag TAG --fileset FILESET

if [ "$EOSDIR" != "None" ]; then
    xrdcp -r condor/out/ EOSDIR
fi