import json
import correctionlib
import importlib.resources
import awkward as ak
import numpy as np
import pickle as pkl
from coffea import processor, hist, util
from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper
from coffea.lookup_tools.dense_lookup import dense_lookup


# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"

POG_JSONS = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
}


# ---------------------------------
# pileup
# ---------------------------------
class PileUPCorrector:
    
    @staticmethod
    def get_UL_year(year):
        if year == "2016":
            year = "2016postVFP"
        elif year == "2016APV":
            year = "2016preVFP"
        return f"{year}_UL"
    
    @staticmethod
    def get_pog_json(year):
        try:
            pog_json = POG_JSONS["pileup"]
        except:
            print(f"No json for pileup")
        year = get_UL_year(year)
        return f"{POG_CORRECTION_PATH}POG/{pog_json[0]}/{year}/{pog_json[1]}"
    
    def add_pileup_weight(self, weights, year, mod, nPU):
        """
        add pileup weight
        
        https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/LUMI_puWeights_Run2_UL/
        """
        cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year + mod))

        year_to_corr = {
            "2016": "Collisions16_UltraLegacy_goldenJSON",
            "2017": "Collisions17_UltraLegacy_goldenJSON",
            "2018": "Collisions18_UltraLegacy_goldenJSON",
        }

        values = {}
        values["nominal"] = cset[year_to_corr[year]].evaluate(nPU, "nominal")
        values["up"] = cset[year_to_corr[year]].evaluate(nPU, "up")
        values["down"] = cset[year_to_corr[year]].evaluate(nPU, "down")

        # add weights (for now only the nominal weight)
        weights.add("pileup", values["nominal"], values["up"], values["down"])
        
        
        
# ---------------------------------
# b-tagging
# ---------------------------------
# b-tag workin points
with open("/home/cms-jovyan/b_lepton_met/data/btagWPs.json", "rb") as handle:
    btagWPs = json.load(handle)

class BTagEfficiency(processor.ProcessorABC):
    def __init__(self, year="2017"):
        self._year = year
        self._accumulator = hist.Hist(
            "Events",
            hist.Cat("tagger", "Tagger"),
            hist.Bin("passWP", "passWP", 2, 0, 2),
            hist.Bin("flavor", "Jet hadronFlavour", [0, 4, 5]),
            hist.Bin("pt", "Jet pT", 20, 40, 300),
            hist.Bin("abseta", "Jet abseta", 4, 0, 2.5),
        )

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        jets = events.Jet[
            (events.Jet.pt > 30.0)
            & (abs(events.Jet.eta) < 2.5)
            & events.Jet.isTight
            & (events.Jet.puId > 0)
        ]
       
        
        out = self.accumulator.identity()
        tags = [
            ("deepJet", "btagDeepFlavB", "M"),
        ]

        for tagger, branch, wp in tags:
            passbtag = jets[branch] > btagWPs[tagger][self._year][wp]

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


class BTagCorrector:
    def __init__(self, wp, tagger="deepJet", year="2017", mod=""):
        self._year = year + mod
        self._tagger = tagger
        self._wp = wp
        self._branch = "btagDeepFlavB"
        self._btagwp = btagWPs[tagger][year + mod][wp]
        

        # more docs at https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/BTV_btagging_Run2_UL/BTV_btagging_201*_UL.html
        if year == "2016":
            self._cset = correctionlib.CorrectionSet.from_file(
                "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016postVFP_UL/btagging.json.gz"
            )
        elif year == "2016APV":
            self._cset = correctionlib.CorrectionSet.from_file(
                "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/2016preVFP_UL/btagging.json.gz"
            )
        else:
            self._cset = correctionlib.CorrectionSet.from_file(
                f"/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/BTV/{year}_UL/btagging.json.gz"
            )
            
        # efficiency lookup
        self.efflookup = util.load(
            f"b_lepton_met/analysis/corrections/data/btageff_{self._tagger}_{self._wp}_{self._year}.coffea"
        )

    def lighttagSF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset["%s_incl" % self._tagger].evaluate(
            syst,
            self._wp,
            np.array(j.hadronFlavour),
            np.array(abs(j.eta)),
            np.array(j.pt),
        )
        return ak.unflatten(sf, nj)

    def btagSF(self, j, syst="central"):
        # syst: central, down, down_correlated, down_uncorrelated, up, up_correlated
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(j), ak.num(j)
        sf = self._cset["%s_comb" % self._tagger].evaluate(
            syst,
            self._wp,
            np.array(j.hadronFlavour),
            np.array(abs(j.eta)),
            np.array(j.pt),
        )
        return ak.unflatten(sf, nj)

    def addBtagWeight(self, jets, weights, label=""):
        """
        Adding one common multiplicative SF (including bcjets + lightjets)
        weights: weights class from coffea
        jets: jets selected in your analysis
        """

        lightJets = jets[(jets.hadronFlavour == 0) & (abs(jets.eta) < 2.5)]
        bcJets = jets[(jets.hadronFlavour > 0) & (abs(jets.eta) < 2.5)]

        lightEff = self.efflookup(
            lightJets.hadronFlavour, lightJets.pt, abs(lightJets.eta)
        )
        bcEff = self.efflookup(bcJets.hadronFlavour, bcJets.pt, abs(bcJets.eta))

        lightPass = lightJets[self._branch] > self._btagwp
        bcPass = bcJets[self._branch] > self._btagwp

        def combine(eff, sf, passbtag):
            # tagged SF = SF*eff / eff = SF
            tagged_sf = ak.prod(sf[passbtag], axis=-1)
            # untagged SF = (1 - SF*eff) / (1 - eff)
            untagged_sf = ak.prod(((1 - sf * eff) / (1 - eff))[~passbtag], axis=-1)

            return ak.fill_none(tagged_sf * untagged_sf, 1.0)

        lightweight = combine(
            lightEff, self.lighttagSF(lightJets, "central"), lightPass
        )
        bcweight = combine(bcEff, self.btagSF(bcJets, "central"), bcPass)

        # nominal weight = btagSF (btagSFbc*btagSFlight)
        nominal = lightweight * bcweight
        weights.add("btagSF" + label, nominal)