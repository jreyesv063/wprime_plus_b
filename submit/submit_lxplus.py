import os
import subprocess
from pathlib import Path
from submit.utils import get_filesets


def run_lxplus(args):
    """
    submit jobs at lxplus using HTCondor

    Arguments:
    -----------
        processor:
            processor to be used {trigger, ttbar, candle, btag_eff}
        executor:
            executor to be used {iterative, futures}
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
            redirector to find CMS datasets {'cmsxrootd.fnal.gov', 'xrootd-cms.infn.it', 'cms-xrd-global.cern.ch'}
        eos:
            wheter to copy or not output files to EOS (default False)
    """
    main_dir = Path.cwd()
    condor_dir = Path(f"{main_dir}/condor")

    # create logs and output directories
    log_dir = Path(str(condor_dir) + "/logs")
    if not log_dir.exists():
        log_dir.mkdir()
        
    # define output directories
    out_dir = Path(f"{condor_dir}/out/")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        
    # define EOS area to move output files
    if args.eos:
        username = os.environ["USER"]
        eos_dir = Path(f"/eos/user/{username[0]}/{username}")
    else:
        eos_dir = None
        
    # divide filesets in args.nsplit json files
    filesets = get_filesets(args.fileset, args.sample, args.year, args.nsplit)

    # submit condor jobs
    for sample, fileset in filesets.items():
        print(f"submitting {sample}")
        
        # create sample directory
        sample_out_dir = Path(f"{out_dir}/{sample}")
        if not sample_out_dir.exists():
            sample_out_dir.mkdir(parents=True)
            
        # make condor file
        condor_template_file = open(f"{condor_dir}/submit.sub")
        local_condor = f"{condor_dir}/{sample}.sub"
        condor_file = open(local_condor, "w")
        for line in condor_template_file:
            line = line.replace("DIRECTORY", str(condor_dir))
            line = line.replace("PREFIX", sample)
            condor_file.write(line)
        condor_file.close()
        condor_template_file.close()

        # make executable file
        sh_template_file = open(f"{condor_dir}/submit.sh")
        local_sh = f"{condor_dir}/{sample}.sh"
        sh_file = open(local_sh, "w")
        for line in sh_template_file:
            line = line.replace("MAINDIRECTORY", str(main_dir))
            line = line.replace("PROCESSOR", args.processor)
            line = line.replace("EXECUTOR", args.executor)
            line = line.replace("WORKERS", str(args.workers))
            line = line.replace("YEAR", args.year)
            line = line.replace("CHANNEL", args.channel)
            line = line.replace("NFILES", str(args.nfiles))
            line = line.replace("REDIRECTOR", args.redirector)
            line = line.replace("OUTPUTLOCATION", str(out_dir))
            line = line.replace("TAG", args.tag)
            line = line.replace("FILESET", fileset)
            line = line.replace("EOSDIR", str(eos_dir))
            sh_file.write(line)
        sh_file.close()
        sh_template_file.close()

        # submit jobs
        subprocess.run(["condor_submit", local_condor])