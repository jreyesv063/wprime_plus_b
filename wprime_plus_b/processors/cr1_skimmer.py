import json
import hist
import pickle
import numpy as np
import awkward as ak
from coffea import processor
from coffea.analysis_tools import Weights, PackedSelection
from .utils import normalize
from .corrections import (
    BTagCorrector,
    add_pileup_weight,
    add_electronID_weight,
    add_electronReco_weight,
    add_electronTrigger_weight,
    add_muon_weight,
    add_muonTriggerIso_weight,
    get_met_corrections,
    get_jec_jer_corrections,
)


class TTbarCR1Skimmer(processor.ProcessorABC):
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
            
        # variables will be store in out and then we'll put them into a column accumulator in output
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
        good_electron_pt = 55 if self._channel == "ele" else 30
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
            corrected_jets, met = get_jec_jer_corrections(
                events, self._year + self._yearmod
            )
        else:
            corrected_jets, met = events.Jet, events.MET
            
        # select good bjets
        good_bjets = (
            (corrected_jets.pt >= 20)
            & (corrected_jets.jetId == 6)
            & (corrected_jets.puId == 7)
            & (corrected_jets.btagDeepFlavB > self._btagDeepFlavB)
            & (np.abs(corrected_jets.eta) < 2.4)
        )
        n_good_bjets = ak.sum(good_bjets, axis=1)
        bjets = corrected_jets[good_bjets]

        # apply MET phi corrections
        met_pt, met_phi = get_met_corrections(
            year=self._year,
            is_mc=self.is_mc,
            met_pt=met.pt,
            met_phi=met.phi,
            npvs=events.PV.npvs,
            mod=self._yearmod,
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
        subleading_bjets = ak.pad_none(bjets, 2)[:, 1]

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

        # cross cleaning: check that leading and subleading bjets does not overlap with our selected leptons
        self.selections.add(
            "lepton_leadingbjet_dr", leading_bjets.delta_r(leptons) > 0.4
        )
        self.selections.add(
            "lepton_subleadingbjet_dr", subleading_bjets.delta_r(leptons) > 0.4
        )

        # add number of leptons and bjets
        self.selections.add("two_bjets", n_good_bjets == 2)
        self.selections.add("one_electron", n_good_electrons == 1)
        self.selections.add("electron_veto", n_good_electrons == 0)
        self.selections.add("one_muon", n_good_muons == 1)
        self.selections.add("muon_veto", n_good_muons == 0)
        self.selections.add("tau_veto", n_good_taus == 0)

        # define selection regions for each channel
        regions = {
            "ele": [
                "lumi",
                "metfilters",
                "trigger_ele",
                "met_pt",
                "two_bjets",
                "tau_veto",
                "muon_veto",
                "one_electron",
                "lepton_leadingbjet_dr",
                "lepton_subleadingbjet_dr",
            ],
            "mu": [
                "lumi",
                "metfilters",
                "trigger_mu",
                "met_pt",
                "two_bjets",
                "tau_veto",
                "electron_veto",
                "one_muon",
                "lepton_leadingbjet_dr",
                "lepton_subleadingbjet_dr",
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
            # pileup reweighting
            add_pileup_weight(
                weights=self.weights,
                year=self._year,
                mod=self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )
            # add btagging weights using the deepJet tagger and the medium working point
            btag_corrector = BTagCorrector(
                wp="M", tagger="deepJet", year=self._year, mod=self._yearmod
            )
            btag_corrector.add_btag_weight(jets=bjets, weights=self.weights)

            # add electron ID and reco weights
            add_electronID_weight(
                weights=self.weights,
                electrons=electrons,
                year=self._year,
                mod=self._yearmod,
                wp="wp80noiso" if self._channel == "ele" else "wp90noiso",
            )
            add_electronReco_weight(
                weights=self.weights,
                electrons=electrons,
                year=self._year,
                mod=self._yearmod,
            )

            if self._channel == "ele":
                # add electron trigger weights
                add_electronTrigger_weight(
                    weights=self.weights,
                    electrons=electrons,
                    year=self._year,
                    mod=self._yearmod,
                )
            # add muon ID and iso weights
            add_muon_weight(
                weights=self.weights,
                muons=muons,
                sf_type="id",
                year=self._year,
                mod=self._yearmod,
                wp="tight" if self._channel == "ele" else "tight",
            )
            add_muon_weight(
                weights=self.weights,
                muons=muons,
                sf_type="iso",
                year=self._year,
                mod=self._yearmod,
                wp="tight" if self._channel == "ele" else "tight",
            )
            if self._channel == "mu":
                # add muon trigger weights
                add_muonTriggerIso_weight(
                    weights=self.weights,
                    muons=muons,
                    year=self._year,
                    mod=self._yearmod,
                )
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