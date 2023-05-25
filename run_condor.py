import os
import sys
import json
import pickle
import argparse
import importlib.resources
from pathlib import Path
from coffea import processor
from datetime import datetime
from wprime_plus_b.processors.candle_processor import CandleProcessor
from wprime_plus_b.processors.signal_processor import SignalRegionProcessor
from wprime_plus_b.processors.ttbar_processor import TTbarControlRegionProcessor
from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
from wprime_plus_b.processors.trigger_efficiency_processor import TriggerEfficiencyProcessor


def main(args):
    # load and process fileset
    fileset = {}
    with open(args.fileset, "r") as handle:
        data = json.load(handle)
    for sample, val in data.items():
        if args.nfiles != -1:
            val = val[: args.nfiles]
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
    }
    executor_args = {
        "schema": processor.NanoAODSchema,
    }
    if args.executor == "futures":
        executor_args.update({"workers": args.workers})
    # run processor
    out = processor.run_uproot_job(
        fileset,
        treename="Events",
        processor_instance=processors[args.processor](**processor_kwargs),
        executor=executors[args.executor],
        executor_args=executor_args,
    )
    # save output
    date = datetime.today().strftime("%Y-%m-%d")
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
    with open(f"{str(output_path)}/{sample}.pkl", "wb") as handle:
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
        "--tag",
        dest="tag",
        type=str,
        default="test",
        help="tag of the submitted jobs",
    )
    parser.add_argument(
        "--fileset",
        dest="fileset",
        type=str,
        help="json fileset",
    )
    args = parser.parse_args()
    main(args)
