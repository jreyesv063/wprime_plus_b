import json
import awkward as ak
import hist as hist2
import importlib.resources
from coffea import processor


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
        
        self.make_output = lambda: hist2.Hist(
            hist2.axis.StrCategory([], name="tagger", growth=True),
            hist2.axis.Regular(20, 20, 500, name="pt"),
            hist2.axis.Regular(4, 0, 2.5, name="abseta", label="Jet abseta"),
            hist2.axis.IntCategory([0, 4, 5], name="flavor"),
            hist2.axis.Regular(2, 0, 2, name="passWP"),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        phasespace_cuts = (
            (abs(events.Jet.eta) < 2.5)
            & (events.Jet.pt > 20.)
        )
        jets = events.Jet[phasespace_cuts]

        out = self.make_output()
        tags = [
            ("deepJet", "btagDeepFlavB", "M"),
        ]
        for tagger, branch, wp in tags:
            passbtag = jets[branch] > self._btagwp
            out.fill(
                tagger=tagger,
                pt=ak.flatten(jets.pt),
                abseta=ak.flatten(abs(jets.eta)),
                flavor=ak.flatten(jets.hadronFlavour),
                passWP=ak.flatten(passbtag),
            )
        return out

    def postprocess(self, a):
        return a