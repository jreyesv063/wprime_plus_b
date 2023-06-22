import json
import dask     
import pickle
import argparse
from pathlib import Path
from coffea import processor
from datetime import datetime
from dask.distributed import Client
from distributed.diagnostics.plugin import UploadDirectory
from wprime_plus_b.processors.candle_processor import CandleProcessor
from wprime_plus_b.processors.signal_processor import SignalRegionProcessor
from wprime_plus_b.processors.ttbar_cr1_processor import TTbarCR1Processor
from wprime_plus_b.processors.ttbar_cr2_processor import TTbarCR2Processor
from wprime_plus_b.processors.btag_efficiency_processor import BTagEfficiencyProcessor
from wprime_plus_b.processors.trigger_efficiency_processor import TriggerEfficiencyProcessor
from wprime_plus_b.processors.cr1_skimmer import TTbarCR1Skimmer
from wprime_plus_b.processors.cr2_skimmer import TTbarCR2Skimmer
from wprime_plus_b.processors.ztoll_skimmer import ZToLLSkimmer

def main(args):
    # load and process filesets
    fileset = {}
    with open(args.fileset, "r") as handle:
        data = json.load(handle)
    for sample, val in data.items():
        if args.nfiles != -1:
            val = val[: args.nfiles]
        fileset[sample] = [f"root://{args.redirector}/" + file for file in val]
    # define processors
    processors = {
        "ttbar_cr1": TTbarCR1Processor,
        "ttbar_cr2": TTbarCR2Processor,
        "ttbar_cr1_skimmer": TTbarCR1Skimmer,
        "ttbar_cr2_skimmer": TTbarCR2Skimmer,
        "ztoll": ZToLLSkimmer,
        "candle": CandleProcessor,
        "trigger": TriggerEfficiencyProcessor,
        "signal": SignalRegionProcessor,
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
        client = Client(args.client)
        executor_args.update({"client": client})
        # upload local directory to dask workers
        try:
            client.register_worker_plugin(
                UploadDirectory(f"{Path.cwd()}", restart=True, update_path=True),
                nanny=True,
            )
            print(f"Uploaded {Path.cwd()} succesfully")
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
        default="ttbar_cr1",
        help="processor to be used {trigger, ttbar_cr1, ttbar_cr2, candle, btag_eff}",
    )
    parser.add_argument(
        "--executor",
        dest="executor",
        type=str,
        default="iterative",
        help="executor to be used {iterative, futures, dask}",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        default="ele",
        help="lepton channel to be processed {'mu', 'ele'}",
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
        help="number of .root files to be processed by sample (default 1. To run all files use -1)",
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
        help="redirector to find CMS datasets {use 'xcache' at coffea-casa. Use 'cmsxrootd.fnal.gov', 'xrootd-cms.infn.it' or 'cms-xrd-global.cern.ch' at lxplus}",
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
    parser.add_argument(
        "--client",
        dest="client",
        type=str,
        help="dask client to use with dask executor on coffea-casa",
    )
    args = parser.parse_args()
    main(args)