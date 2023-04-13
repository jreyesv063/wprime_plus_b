import json
import hist 
import importlib.resources
import awkward as ak
from coffea import processor
from coffea.analysis_tools import Weights

class BTagEfficiencyProcessor(processor.ProcessorABC):
    """
    Compute btag efficiencies for a tagger in a given working point

    Parameters:
    -----------
        year:
            year of the MC samples
        yearmod:
            year modifier {"", "APV"} (use "APV" for pre 2016 datasets)
        tagger:
            tagger name {'deepJet', 'deepCSV'}
        wp:
            worging point {'L', 'M', 'T'}
    """
    def __init__(self, year="2017", yearmod="", tagger="deepJet", wp="M"):
        self._year = year + yearmod
        self._tagger = tagger
        self._wp = wp
        
        with importlib.resources.path("wprime_plus_b.data", "btagWPs.json") as path:
            with open(path, "r") as handle:
                btagWPs = json.load(handle)
        self._btagwp = btagWPs[self._tagger][self._year][self._wp]
        
        self.make_output = lambda: hist.Hist(
            hist.axis.Regular(20, 20, 500, name="pt"),
            hist.axis.Regular(4, 0, 2.5, name="abseta"),
            hist.axis.IntCategory([0, 4, 5], name="flavor"),
            hist.axis.Regular(2, 0, 2, name="passWP"),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        out = self.make_output()
        
        phasespace_cuts = (
            (abs(events.Jet.eta) < 2.5)
            & (events.Jet.pt > 20.)
        )
        jets = events.Jet[phasespace_cuts]
        passbtag = jets.btagDeepFlavB > self._btagwp
        
        out.fill(
            pt=ak.flatten(jets.pt),
            abseta=ak.flatten(abs(jets.eta)),
            flavor=ak.flatten(jets.hadronFlavour),
            passWP=ak.flatten(passbtag),
        )
        return {dataset: out}

    def postprocess(self, accumulator):
        return accumulator