import os
from submit.utils import get_filesets


def run_coffea_casa(args):
    """
    submit jobs at coffea-casa

    Arguments:
    -----------
        processor:
            processor to be used {trigger, ttbar, candle, btag_eff}
        executor:
            executor to be used {iterative, futures, dask}
        channel:
            channel to be processed {'mu', 'ele'}
        fileset:
            fileset to be processed (use 'UL' to select all UL samples)
        sample:
            sample key to be processed {'all', 'mc' or <sample_name>}
        year:
            year of the data {2016, 2017, 2018}
        yearmod:
            year modifier {'', 'APV'}
        workers:
            number of workers to use with futures executor
        nsplit:
            number of subsets to divide the fileset into
        nfiles:
            number of .root files to be processed by sample
        tag:
            tag of the submitted jobs
        redirector:
            redirector to find CMS datasets. 'xcache' at coffea-casa. 'cmsxrootd.fnal.gov', 'xrootd-cms.infn.it' or 'cms-xrd-global.cern.ch' at lxplus.
    """
    # dask client at coffea-casa
    client = "tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786"

    # divide filesets in args.nsplit json files
    filesets = get_filesets(args.fileset, args.sample, args.year, args.nsplit)

    # submit jobs
    if len(args.nsample) == 0:
        for sample, fileset in filesets.items():
            print(f"Processing {sample}")
            os.system(
                f"python3 run.py --processor {args.processor} --executor {args.executor} --channel {args.channel} --fileset {fileset} --year {args.year} --nfiles {args.nfiles} --tag {args.tag} --redirector {args.redirector} --client {client}"
            )
    else:
        for sample, fileset in filesets.items():
            for n in args.nsample:
                if f"_{n}" in sample:
                    print(f"Processing {sample}")
                    os.system(
                        f"python3 run.py --processor {args.processor} --executor {args.executor} --channel {args.channel} --fileset {fileset} --year {args.year} --nfiles {args.nfiles} --tag {args.tag} --redirector {args.redirector} --client {client}"
                    )
                    