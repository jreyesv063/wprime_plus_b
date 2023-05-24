import os
import sys
import json
import pickle
import argparse
import dask
import importlib.resources
from pathlib import Path
from coffea import processor
from datetime import datetime
from dask.distributed import Client
from distributed.diagnostics.plugin import UploadDirectory
from wprime_plus_b.processors.candle_processor import CandleProcessor
from wprime_plus_b.processors.signal_processor import SignalRegionProcessor
from wprime_plus_b.processors.ttbar_processor import TTbarControlRegionProcessor
from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
from wprime_plus_b.processors.trigger_efficiency_processor import TriggerEfficiencyProcessor


def main(args):
    """
    # load simplified names for datasets
    with importlib.resources.path(
        "wprime_plus_b.data", "simplified_samples.json"
    ) as path:
        with open(path, "r") as handle:
            simplified_samples = json.load(handle)[args.year]
            simplified_samples_r = {v: k for k, v in simplified_samples.items()}

    # define fileset
    def get_data(fileset):
        with importlib.resources.path("wprime_plus_b.fileset", fileset) as path:
            with open(path, "r") as handle:
                data = json.load(handle)
        return data

    is_mc = all(s not in args.sample for s in ("Electron", "Muon"))
    is_ul = args.sample in simplified_samples_r # is a full UL sample?
    is_candle = args.processor == "candle"

    if is_ul:
        data = get_data(f"fileset_{args.year}_UL_NANO.json")
    elif is_candle and "DYJets" in args.sample:
        data = get_data("fileset_candle.json")
    else:
        data = get_data(f"{args.sample}.json")

    fileset = {}
    for key, val in data.items():
        sample = simplified_samples[key] if key in simplified_samples else args.sample
        if args.nfiles != -1:
            val = val[: args.nfiles]
        if not args.run_all and args.sample not in key:
            continue
        fileset[sample] = [f"root://{args.redirector}/" + file for file in val]

    # define processors
    processors = {
        "ttbar": TTbarControlRegionProcessor,
        "trigger": TriggerEfficiencyProcessor,
        "signal": SignalRegionProcessor,
        "candle": CandleProcessor,
        "btag_eff": BTagEfficiencyProcessor,
    }
    processor_kwargs = {
        "year": args.year,
        "yearmod": args.yearmod,
        "channel": args.channel,
    }
    if args.processor == "btag_eff":
        del processor_kwargs["channel"]

    # define executors
    executors = {
        "iterative": processor.iterative_executor,
        "futures": processor.futures_executor,
        "dask": processor.dask_executor,
    }
    executor_args = {
        "schema": processor.NanoAODSchema,
    }
    if args.executor == "futures":
        executor_args.update({"workers": args.workers})
    if args.executor == "dask":
        loc_base = os.environ["PWD"]
        client = Client(args.client)
        executor_args.update({"client": client})
        # upload local directory to dask workers
        try:
            client.register_worker_plugin(
                UploadDirectory(f"{loc_base}", restart=True, update_path=True),
                nanny=True,
            )
            print(f"Uploaded {loc_base} succesfully")
        except OSError:
            print("Failed to upload the directory")

    # run processor
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=processors[args.processor](**processor_kwargs),
        executor=executors[args.executor],
        executor_args=executor_args,
    )
    """
    # save output
    date = datetime.today().strftime("%Y-%m-%d")
    tag = args.processor if args.run_all else args.sample
    output_path = Path(
        args.output_location
        + "/"
        + args.tag
        + "/"
        + date
        + "/"
        + args.processor
        + "/"
        + args.year
        + "/"
        + args.channel
    )
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    with open(f'{str(output_path)}/{args.sample}.pkl', 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
        help="executor {iterative, futures, dask}",
    )
    parser.add_argument(
        "--sample",
        dest="sample",
        type=str,
        default="TTTo2L2Nu",
        help="name of sample to process",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="ele",
        help="lepton channel {ele, mu}",
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
        "--client",
        dest="client",
        type=str,
        default="tls://daniel-2eocampo-2ehenao-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786",
        help="dask client to use with dask executor",
    )
    parser.add_argument(
        "--redirector",
        dest="redirector",
        type=str,
        default="xcache",
        help="redirector to acces data {xcache to use at coffea-casa}",
    )
    parser.add_argument(
        "--output_location",
        dest="output_location",
        type=str,
        default="./outfiles/",
        help="output location (default ./outfiles)",
    )
    parser.add_argument(
        "--run_all",
        dest="run_all",
        type=bool,
        default=False,
        help="if True run all datasets in fileset",
    )
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        default="test",
        help="tag of the submitted jobs",
    )

    args = parser.parse_args()
    main(args)
