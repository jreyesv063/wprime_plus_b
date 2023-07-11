import json
import gzip
import cloudpickle
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type, Tuple
from coffea.lookup_tools import extractor
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.base import NanoEventsArray


# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"

# summary of pog scale factors: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
POG_JSONS = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btag": ["BTV", "btagging.json.gz"],
    "met": ["JME", "met.json.gz"],
}

pog_years = {
    "2016": "2016postVFP_UL",
    "2016APV": "2016preVFP_UL",
    "2017": "2017_UL",
    "2018": "2018_UL",
}


def get_pog_json(json_name: str, year: str) -> str:
    """
    returns the path to the pog json file

    Parameters:
    -----------
        json_name:
            json name {'muon', 'electron', 'pileup', 'btag'}
        year:
            dataset year {'2016', '2017', '2018'}
    """
    if json_name in POG_JSONS:
        pog_json = POG_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    return f"{POG_CORRECTION_PATH}/POG/{pog_json[0]}/{pog_years[year]}/{pog_json[1]}"


def clip_array(array: ak.Array, target=2, fill_value=1) -> ak.Array:
    """
    Clips an awkward array to a fixed length by padding with None values. Fills any remaining
    None values in the clipped array with 'fill_value'

    Parameters:
    -----------
        array:
            Data containing nested lists to pad to a target length
        target:
            The intended length of the lists. The output lists will have exactly this length.
        fill_value:
            The value used to fill any remaining None values after clipping. Defaults to 1.

    Returns:
        ak.Array: The clipped collection with a fixed length, padded with None values and filled as specified.
    """
    return ak.fill_none(ak.pad_none(array, target, clip=True), fill_value)