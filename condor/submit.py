import os
import json
import argparse
import subprocess
from pathlib import Path
from utils import divide_fileset


def main(args):
    main_dir = Path.cwd()
    condor_dir = Path(f"{main_dir}/condor")

    # create logs and output directories
    log_dir = Path(str(condor_dir) + "/logs")
    if not log_dir.exists():
        log_dir.mkdir()
        
    # define EOS and output directories
    username = os.environ["USER"]
    eos_dir = Path(f"/eos/user/{username[0]}/{username}")
    out_dir = Path(f"{condor_dir}/out/")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        
    # create args.nsplit json files for each sample
    with open(f"{main_dir}/wprime_plus_b/data/simplified_samples.json", "r") as handle:
        simplified_samples = json.load(handle)[args.year]
    samples = list(simplified_samples.keys())
    filesets = divide_fileset(samples, args.nsplit, args.year)
    if args.sample != "all":
        filesets = {key: filesets[key] for key in filesets if args.sample in key}
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor",
        dest="processor",
        type=str,
        default="ttbar",
        help="processor to run {trigger, ttbar, signal, candle, btag_eff}",
    )
    parser.add_argument(
        "--executor",
        dest="executor",
        type=str,
        default="iterative",
        help="executor {iterative, futures}",
    )
    parser.add_argument("--year", dest="year", type=str, default="2017", help="year")
    parser.add_argument(
        "--yearmod",
        dest="yearmod",
        type=str,
        default="",
        help="year modifier {'', 'APV'}",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="mu",
        help="lepton channel {ele, mu}",
    )
    parser.add_argument(
        "--nfiles",
        dest="nfiles",
        type=int,
        default=1,
        help="number of files per sample (default 1. To run all files use -1)",
    )
    parser.add_argument(
        "--workers",
        dest="workers",
        type=int,
        default=4,
        help="number of workers to use with futures executor (default 4)",
    )
    parser.add_argument(
        "--redirector",
        dest="redirector",
        type=str,
        default="cmsxrootd.fnal.gov",
        help="redirector to acces CMS datasets",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="test",
        help="tag of the submitted jobs",
    )
    parser.add_argument(
        "--nsplit",
        dest="nsplit",
        type=str,
        default=5,
        help="number of splits for each sample",
    )
    parser.add_argument(
        "--sample",
        dest="sample",
        type=str,
        default="all",
        help="name of the sample (see simplified_samples values)",
    )
    main(parser.parse_args())
