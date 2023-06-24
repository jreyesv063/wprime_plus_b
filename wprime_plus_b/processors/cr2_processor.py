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


class TTbarCR2Processor(processor.ProcessorABC):
    def __init__(
        self,
        year: str = "2017",
        yearmod: str = "",
        channel: str = "ele",
    ):
        self._year = year
        self._yearmod = yearmod
        self._channel = channel

        # open and load triggers
        with open("wprime_plus_b/data/triggers.json", "r") as f:
            self._triggers = json.load(f)[self._year]
        # open and load btagDeepFlavB working points
        with open("wprime_plus_b/data/btagDeepFlavB.json", "r") as f:
            self._btagDeepFlavB = json.load(f)[self._year]
        # open and load met filters
        with open("wprime_plus_b/data/metfilters.json", "rb") as handle:
            self._metfilters = json.load(handle)[self._year]
        # open and load lumi masks
        with open("wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
            self._lumi_mask = pickle.load(handle)
            
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
        good_electron_pt = 55 if self._channel == "mu" else 30
        good_electrons = (
            (events.Electron.pt >= good_electron_pt)
            & (np.abs(events.Electron.eta) < 2.4)
            & (
                (np.abs(events.Electron.eta) < 1.44)
                | (np.abs(events.Electron.eta) > 1.57)
            )
            & (
                events.Electron.mvaFall17V2Iso_WP80
                if self._channel == "ele"
                else events.Electron.mvaFall17V2Iso_WP90
            )
            & (
                events.Electron.pfRelIso04_all < 0.25
                if hasattr(events.Electron, "pfRelIso04_all")
                else events.Electron.pfRelIso03_all < 0.25
            )
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)
        electrons = events.Electron[good_electrons]

        # select good muons
        good_muons = (
            (events.Muon.pt >= 35)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.tightId)
            & (
                events.Muon.pfRelIso04_all < 0.25
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.25
            )
        )
        n_good_muons = ak.sum(good_muons, axis=1)
        muons = events.Muon[good_muons]

        # select good taus
        good_taus = (
            (events.Tau.idDeepTau2017v2p1VSjet > 8)
            & (events.Tau.idDeepTau2017v2p1VSe > 8)
            & (events.Tau.idDeepTau2017v2p1VSmu > 1)
            & (np.abs(events.Tau.eta) < 2.3)
            & (events.Tau.pt > 20)
            & (events.Tau.dz < 0.2)
        )
        n_good_taus = ak.sum(good_taus, axis=1)
        taus = events.Tau[good_taus]

        # apply JEC/JER corrections to MC jets (propagate corrections to MET)
        # in data, the corrections are already applied
        if self.is_mc:
            jets, met = jet_corrections(events, self._year + self._yearmod)
        else:
            jets, met = events.Jet, events.MET
            
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

        # apply MET phi corrections
        met_pt, met_phi = met_phi_corrections(
            met_pt=met.pt,
            met_phi=met.phi,
            npvs=events.PV.npvs,
            is_mc=self.is_mc,
            year=self._year,
            year_mod=self._yearmod,
        )
        met["pt"], met["phi"] = met_pt, met_phi

        # ---------------
        # event variables
        # ---------------
        # We can define the variables for leptons from just the leading (in pt) lepton
        # since all of our signal and control regions require exactly zero or one of
        # them so there is no ambiguity to resolve.
        leptons = ak.firsts(electrons) if self._channel == "ele" else ak.firsts(muons)

        # Some control regions require more than one bjet though, however we will
        # compute all variables using the leading bjet
        leading_bjets = ak.firsts(bjets)

        # lepton relative isolation
        lepton_reliso = (
            leptons.pfRelIso04_all
            if hasattr(leptons, "pfRelIso04_all")
            else leptons.pfRelIso03_all
        )
        # lepton-bjet deltaR and invariant mass
        lepton_bjet_dr = leading_bjets.delta_r(leptons)
        lepton_bjet_mass = (leptons + leading_bjets).mass

        # lepton-MET transverse mass and deltaPhi
        lepton_met_mass = np.sqrt(
            2.0
            * leptons.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(leptons.delta_phi(met)))
        )
        lepton_met_delta_phi = np.abs(leptons.delta_phi(met))

        # lepton-bJet-MET total transverse mass
        lepton_met_bjet_mass = np.sqrt(
            (leptons.pt + leading_bjets.pt + met.pt) ** 2
            - (leptons + leading_bjets + met).pt ** 2
        )
        # add variables to out
        self.add_var("lepton_pt", leptons.pt)
        self.add_var("lepton_eta", leptons.eta)
        self.add_var("lepton_phi", leptons.phi)
        self.add_var("jet_pt", leading_bjets.pt)
        self.add_var("jet_eta", leading_bjets.eta)
        self.add_var("jet_phi", leading_bjets.phi)
        self.add_var("met", met_pt)
        self.add_var("met_phi", met_phi)
        self.add_var("lepton_bjet_dr", lepton_bjet_dr)
        self.add_var("lepton_bjet_mass", lepton_bjet_mass)
        self.add_var("lepton_met_mass", lepton_met_mass)
        self.add_var("lepton_met_delta_phi", lepton_met_delta_phi)
        self.add_var("lepton_met_bjet_mass", lepton_met_bjet_mass)

        # ---------------
        # event selection
        # ---------------
        # make a PackedSelection object to manage the event selections easily
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

        # check that there be a minimum MET greater than 50 GeV
        self.selections.add("met_pt", met.pt > 50)

        # cross-cleaning 
        # check that bjet does not overlap with our selected leptons
        self.selections.add(
            "electron_bjet_dr", leading_bjets.delta_r(ak.firsts(electrons)) > 0.4
        )
        self.selections.add(
            "muon_bjet_dr", leading_bjets.delta_r(ak.firsts(muons)) > 0.4
        )
        # check that selected leptons does not overlap
        self.selections.add(
            "electron_muon_dr", ak.firsts(electrons).delta_r(ak.firsts(muons)) > 0.4
        )
        
        # add number of leptons and bjets
        self.selections.add("one_bjet", n_good_bjets == 1)
        self.selections.add("one_electron", n_good_electrons == 1)
        self.selections.add("one_muon", n_good_muons == 1)
        self.selections.add("tau_veto", n_good_taus == 0)

        # define selection regions for each channel
        regions = {
            "ele": [
                "lumi",
                "metfilters",
                "trigger_mu",
                "met_pt",
                "one_bjet",
                "tau_veto",
                "one_muon",
                "one_electron",
                "muon_bjet_dr",
                "electron_bjet_dr",
                "electron_muon_dr"
            ],
            "mu": [
                "lumi",
                "metfilters",
                "trigger_ele",
                "met_pt",
                "one_bjet",
                "tau_veto",
                "one_electron",
                "one_muon",
                "muon_bjet_dr",
                "electron_bjet_dr",
                "electron_muon_dr"
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
            
            if self._channel == "ele":
                # add muon trigger weights
                muon_corrector.add_triggeriso_weight()
            if self._channel == "mu":
                # add electron trigger weights
                electron_corrector.add_trigger_weight()
                
        # save total weight from the weights container
        self.add_var("weights", self.weights.weight())

        # -----------------------------
        # output accumulator definition
        # -----------------------------
        # combine all region selections into a single mask
        selections = regions[self._channel]
        final_cut = self.selections.all(*selections)

        # select variables and put them in column accumulators
        self.output = {
            key: processor.column_accumulator(normalize(val, final_cut))
            for key, val in self.out.items()
        }
        # save sum of gen weights
        if self.is_mc:
            self.output["sumw"] = ak.sum(events.genWeight)
        # save number of events before and after selection
        self.output["events_before"] = nevents
        self.output["events_after"] = ak.sum(final_cut)

        return self.output

    def postprocess(self, accumulator):
        return accumulator