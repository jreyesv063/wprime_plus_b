import json
import pickle
import numpy as np
import awkward as ak
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from .utils import normalize
from .corrections import (
    BTagCorrector,
    ElectronCorrector,
    MuonCorrector,
    add_pileup_weight,
    met_phi_corrections,
    jet_corrections,
)


class ZToLLProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str = "2017",
        yearmod: str = "",
        channel: str = "ele",
    ):
        self._year = year
        self._yearmod = yearmod
        self._channel = channel
        
        # open triggers
        with open("wprime_plus_b/data/triggers.json", "r") as f:
            self._triggers = json.load(f)[self._year]
        # open met filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with open("wprime_plus_b/data/metfilters.json", "rb") as handle:
            self._metfilters = json.load(handle)[self._year]
        # open lumi masks
        with open("wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
            self._lumi_mask = pickle.load(handle)
        # open btagDeepFlavB
        with open("wprime_plus_b/data/btagDeepFlavB.json", "r") as f:
            self._btagDeepFlavB = json.load(f)[self._year]

        # variables will be store in out and then will be put in a column accumulator in output
        self.out = {}
        self.output = {}
        
    def add_var(self, name: str, var: ak.Array):
        """add a variable array to the out dictionary"""
        self.out = {**self.out, name: var}
        
    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        # check if sample is MC
        self.is_mc = hasattr(events, "genWeight")
        
        # get number of events before selection
        nevents = len(events)

        # ------------------
        # event preselection
        # ------------------
        # select good electrons
        good_electrons = (
            (events.Electron.pt >= 40)
            & (np.abs(events.Electron.eta) < 2.4)
            & (
                (np.abs(events.Electron.eta) < 1.44)
                | (np.abs(events.Electron.eta) > 1.57)
            )
            & events.Electron.mvaFall17V2Iso_WP80
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)
        electrons = events.Electron[good_electrons]
        
        # muons
        good_muons = (
            (events.Muon.pt >= 35)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.mediumId)
        )
        n_good_muons = ak.sum(good_muons, axis=1)
        muons = events.Muon[good_muons]
        
        # apply JEC/JER corrections to MC jets (propagate corrections to MET)
        # in data, the corrections are already applied
        if self.is_mc:
            jets, _ = jet_corrections(events, self._year + self._yearmod)
        else:
            jets, _ = events.Jet, events.MET
            
        # select good bjets
        good_bjets = (
            (jets.pt >= 20)
            & (jets.jetId == 6)
            & (jets.puId == 7)
            & (jets.btagDeepFlavB > self._btagDeepFlavB)
            & (np.abs(jets.eta) < 2.4)
        )
        n_good_bjets = ak.sum(good_bjets, axis=1)
        bjets = jets[good_bjets]
        
        # ---------------
        # event variables
        # ---------------
        leptons = electrons if self._channel == "ele" else muons
        n_good_leptons = n_good_electrons if self._channel == "ele" else n_good_muons
        leading_lepton = ak.firsts(leptons)
        subleading_lepton = ak.pad_none(leptons, 2)[:, 1]
        
        # add dilepton invariant mass to out
        dilepton_mass = (leading_lepton + subleading_lepton).mass
        self.add_var("dilepton_mass", dilepton_mass)
        
        # ---------------
        # event selection
        # ---------------
        # make a PackedSelection object to manage selections
        self.selections = PackedSelection()
        
        # add luminosity calibration mask (only to data)
        if not self.is_mc:
            lumi_mask = self._lumi_mask[self._year](events.run, events.luminosityBlock)
        else:
            lumi_mask = np.ones(len(events), dtype="bool")
        self.selections.add("lumi", lumi_mask)

        # add MET filters mask
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.is_mc else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        self.selections.add("metfilters", metfilters)

        # add lepton triggers masks
        trigger = {}
        for ch in ["ele", "mu"]:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            for t in self._triggers[ch]:
                if t in events.HLT.fields:
                    trigger[ch] = trigger[ch] | events.HLT[t]
        self.selections.add("trigger_ele", trigger["ele"])
        self.selections.add("trigger_mu", trigger["mu"])
        
        # check that we have 2l events
        self.selections.add("two_leptons", n_good_leptons == 2)
        # check that leading electron pt is greater than 45 GeV
        self.selections.add("leading_electron", leading_lepton.pt > 45)
        # check that dilepton system is neutral
        self.selections.add("neutral", leading_lepton.charge * subleading_lepton.charge < 0)
        # check that dilepton invariant mass is between 60 and 120 GeV
        self.selections.add("mass_range", (60 < dilepton_mass) & (dilepton_mass < 120))
        # veto bjets
        self.selections.add("bjet_veto", n_good_bjets == 0)
        
        # define selection regions for each channel
        regions = {
            "ele": [
                "lumi",
                "metfilters",
                "trigger_ele",
                "bjet_veto",
                "two_leptons",
                "leading_electron",
                "neutral",
                "mass_range",
            ],
            "mu": [
                "lumi",
                "metfilters",
                "trigger_mu",
                "bjet_veto",
                "two_leptons",
                "neutral",
                "mass_range",
            ],
        }
        
        # -------------
        # event weights
        # -------------
        # define weights container
        self.weights = Weights(nevents, storeIndividual=True)
        if self.is_mc:
            # add gen weigths
            gen_weight = events.genWeight
            self.weights.add("genweight", gen_weight)

            # add L1prefiring weights
            if self._year in ("2016", "2017"):
                self.weights.add(
                    "L1Prefiring",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
            # add pileup reweighting
            add_pileup_weight(
                n_true_interactions=ak.to_numpy(events.Pileup.nPU),
                weights=self.weights,
                year=self._year,
                year_mod=self._yearmod,
            )
            # b-tagging corrector
            btag_corrector = BTagCorrector(
                sf_type="comb",
                worging_point="M",
                tagger="deepJet",
                year=self._year,
                year_mod=self._yearmod,
            )
            # add btagging weights
            btag_corrector.add_btag_weight(jets=bjets, weights=self.weights)

            if self._channel == "ele":
                # electron corrector
                electron_corrector = ElectronCorrector(
                    electrons=ak.firsts(electrons),
                    weights=self.weights,
                    year=self._year,
                    year_mod=self._yearmod,
                )
                # add electron ID weights
                electron_corrector.add_id_weight(
                    working_point="wp80noiso" if self._channel == "ele" else "wp90noiso",
                )
                # add electron reco weights
                electron_corrector.add_reco_weight()
                # add electron trigger weights
                electron_corrector.add_trigger_weight()

            if self._channel == "mu":
                # muon corrector
                muon_corrector = MuonCorrector(
                    muons=ak.firsts(muons),
                    weights=self.weights,
                    year=self._year,
                    year_mod=self._yearmod,
                )
                # add muon ID weights
                muon_corrector.add_id_weight(working_point="tight")
                # add muon iso weights
                muon_corrector.add_iso_weight(working_point="tight")
                # add muon trigger weights
                muon_corrector.add_triggeriso_weight()
                    
        # save total weight from the weights container
        self.add_var("weights", self.weights.weight())
        
        # -----------------------------
        # output accumulator definition
        # -----------------------------
        # combine all region selections into a single mask
        selections = regions[self._channel]
        selections_mask = self.selections.all(*selections)
        
        # select variables and put them in column accumulators
        self.output = {
            key: processor.column_accumulator(normalize(val, selections_mask))
            for key, val in self.out.items()
        }
        # save sum of gen weights
        if self.is_mc:
            self.output["sumw"] = ak.sum(events.genWeight)
        # save number of events before and after selection
        self.output["events_before"] = nevents
        self.output["events_after"] = ak.sum(selections_mask)

        return self.output
    
    def postprocess(self, accumulator):
        return accumulator