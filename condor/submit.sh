#!/bin/bash

export XRD_NETWORKSTACK=IPv4
export X509_USER_PROXY=$HOME/tmp/x509up
cd MAINDIRECTORY

python3 run_condor.py --processor PROCESSOR --executor EXECUTOR --workers WORKERS --sample SAMPLE --channel CHANNEL --nfiles NFILES --year YEAR --redirector REDIRECTOR --output_location OUTPUTLOCATION --tag TAG --fileset FILESET