import json
import pickle
import numpy as np
import pandas as pd
import awkward as ak
import hist as hist2
from typing import List
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
)


class TriggerEfficiencyProcessor(processor.ProcessorABC):
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
        # open btagDeepFlavB
        with open("wprime_plus_b/data/btagDeepFlavB.json", "r") as f:
            self._btagDeepFlavB = json.load(f)[self._year]
        # open met filters
        # https://twiki.cern.ch/twiki/bin/view/CMS/MissingETOptionalFiltersRun2
        with open("wprime_plus_b/data/metfilters.json", "rb") as handle:
            self._metfilters = json.load(handle)[self._year]
        # open lumi masks
        with open("wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
            self._lumi_mask = pickle.load(handle)
        # output histograms
        self.make_output = lambda: {
            "sumw": 0,
            "cutflow": {
                "numerator": {},
                "denominator": {},
            },
            "electron_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="electron_pt",
                    label=r"electron $p_T$ [GeV]",
                ),
                hist2.axis.Regular(
                    25, 0, 1, name="electron_relIso", label="electron RelIso"
                ),
                hist2.axis.Regular(
                    50, -2.4, 2.4, name="electron_eta", label="electron $\eta$"
                ),
                hist2.storage.Weight(),
            ),
            "muon_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="muon_pt",
                    label=r"muon $p_T$ [GeV]",
                ),
                hist2.axis.Regular(25, 0, 1, name="muon_relIso", label="muon RelIso"),
                hist2.axis.Regular(50, -2.4, 2.4, name="muon_eta", label="muon $\eta$"),
                hist2.storage.Weight(),
            ),
            "jet_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="jet_pt",
                    label=r"bJet $p_T$ [GeV]",
                ),
                hist2.axis.Regular(50, -2.4, 2.4, name="jet_eta", label="bJet $\eta$"),
                hist2.storage.Weight(),
            ),
            "met_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [50, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="met_pt",
                    label=r"$p_T^{miss}$ [GeV]",
                ),
                hist2.axis.Regular(
                    50, -4.0, 4.0, name="met_phi", label=r"$\phi(p_T^{miss})$"
                ),
                hist2.storage.Weight(),
            ),
            "electron_bjet_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(
                    30, 0, 5, name="electron_bjet_dr", label="$\Delta R(e, bJet)$"
                ),
                hist2.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="invariant_mass",
                    label=r"$m(e, bJet)$ [GeV]",
                ),
                hist2.storage.Weight(),
            ),
            "muon_bjet_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Regular(
                    30, 0, 5, name="muon_bjet_dr", label="$\Delta R(\mu, bJet)$"
                ),
                hist2.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="invariant_mass",
                    label=r"$m(\mu, bJet)$ [GeV]",
                ),
                hist2.storage.Weight(),
            ),
            "lep_met_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="electron_met_transverse_mass",
                    label=r"$m_T(e, p_T^{miss})$ [GeV]",
                ),
                hist2.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="muon_met_transverse_mass",
                    label=r"$m_T(\mu, p_T^{miss})$ [GeV]",
                ),
                hist2.storage.Weight(),
            ),
            "lep_bjet_met_kin": hist2.Hist(
                hist2.axis.StrCategory([], name="region", growth=True),
                hist2.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="electron_total_transverse_mass",
                    label=r"$m_T^{tot}(e, bJet, p_T^{miss})$ [GeV]",
                ),
                hist2.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="muon_total_transverse_mass",
                    label=r"$m_T^{tot}(\mu, bJet, p_T^{miss})$ [GeV]",
                ),
                hist2.storage.Weight(),
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata["dataset"]
        nevents = len(events)
        self.isMC = hasattr(events, "genWeight")
        self.output = self.make_output()
        self.output["cutflow"]["nevents"] = nevents

        # luminosity
        if not self.isMC:
            lumi_mask = self._lumi_mask[self._year](events.run, events.luminosityBlock)
        else:
            lumi_mask = np.ones(len(events), dtype="bool")
        # MET filters
        metfilters = np.ones(nevents, dtype="bool")
        metfilterkey = "mc" if self.isMC else "data"
        for mf in self._metfilters[metfilterkey]:
            if mf in events.Flag.fields:
                metfilters = metfilters & events.Flag[mf]
        # triggers
        trigger = {}
        for ch in ["ele", "mu"]:
            trigger[ch] = np.zeros(nevents, dtype="bool")
            for t in self._triggers[ch]:
                if t in events.HLT.fields:
                    trigger[ch] = trigger[ch] | events.HLT[t]
        # electrons
        good_electrons = (
            (events.Electron.pt >= 30)
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

        ele_reliso = (
            electrons.pfRelIso04_all
            if hasattr(electrons, "pfRelIso04_all")
            else electrons.pfRelIso03_all
        )
        # muons
        good_muons = (
            (events.Muon.pt >= 30)
            & (np.abs(events.Muon.eta) < 2.4)
            & (events.Muon.mediumId if self._channel == "ele" else events.Muon.tightId)
            & (
                events.Muon.pfRelIso04_all < 0.25
                if hasattr(events.Muon, "pfRelIso04_all")
                else events.Muon.pfRelIso03_all < 0.25
            )
        )
        n_good_muons = ak.sum(good_muons, axis=1)
        muons = events.Muon[good_muons]

        mu_reliso = (
            muons.pfRelIso04_all
            if hasattr(muons, "pfRelIso04_all")
            else muons.pfRelIso03_all
        )
        # b-jets
        good_bjets = (
            (events.Jet.pt >= 20)
            & (np.abs(events.Jet.eta) < 2.4)
            & (events.Jet.jetId == 6)
            & (events.Jet.puId == 7)
            & (events.Jet.btagDeepFlavB > self._btagDeepFlavB)
        )
        n_good_bjets = ak.sum(good_bjets, axis=1)
        candidatebjet = ak.firsts(events.Jet[good_bjets])

        # missing energy
        met = events.MET
        met["pt"], met["phi"] = get_met_corrections(
            year=self._year,
            is_mc=self.isMC,
            met_pt=met.pt,
            met_phi=met.phi,
            npvs=events.PV.npvs,
            mod=self._yearmod,
        )
        # lepton-bjet delta R and invariant mass
        ele_bjet_dr = candidatebjet.delta_r(electrons)
        ele_bjet_mass = (electrons + candidatebjet).mass
        mu_bjet_dr = candidatebjet.delta_r(muons)
        mu_bjet_mass = (muons + candidatebjet).mass

        # lepton-MET transverse mass
        ele_met_tranverse_mass = np.sqrt(
            2.0
            * electrons.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(electrons.delta_phi(met)))
        )
        mu_met_transverse_mass = np.sqrt(
            2.0
            * muons.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(muons.delta_phi(met)))
        )

        # lepton-bJet-MET total transverse mass
        ele_total_transverse_mass = np.sqrt(
            (electrons.pt + candidatebjet.pt + met.pt) ** 2
            - (electrons + candidatebjet + met).pt ** 2
        )
        mu_total_transverse_mass = np.sqrt(
            (muons.pt + candidatebjet.pt + met.pt) ** 2
            - (muons + candidatebjet + met).pt ** 2
        )
        # weights
        weights = Weights(nevents, storeIndividual=True)
        if self.isMC:
            # genweight
            self.output["sumw"] = ak.sum(events.genWeight)
            weights.add("genweight", events.genWeight)
            # L1prefiring
            if self._year in ("2016", "2017"):
                weights.add(
                    "L1Prefiring",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
            # pileup
            add_pileup_weight(
                weights=weights,
                year=self._year,
                mod=self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )
            # b-tagging
            self._btagSF = BTagCorrector(
                wp="M", tagger="deepJet", year=self._year, mod=self._yearmod
            )
            self._btagSF.add_btag_weight(events=events, weights=self.weights)

            # electron weights
            add_electronID_weight(
                weights=weights,
                electrons=electrons,
                year=self._year,
                mod=self._yearmod,
                wp="wp80noiso" if self._channel == "ele" else "wp90noiso",
            )
            add_electronReco_weight(
                weights=weights,
                electrons=electrons,
                year=self._year,
                mod=self._yearmod,
            )
            add_electronTrigger_weight(
                weights=weights,
                electrons=electrons,
                year=self._year,
                mod=self._yearmod,
            )
            # muon weights
            add_muon_weight(
                weights=weights,
                muons=muons,
                sf_type="id",
                year=self._year,
                mod=self._yearmod,
                wp="medium" if self._channel == "ele" else "tight",
            )
            add_muon_weight(
                weights=weights,
                muons=muons,
                sf_type="iso",
                year=self._year,
                mod=self._yearmod,
                wp="medium" if self._channel == "ele" else "tight",
            )
            add_muonTriggerIso_weight(
                weights=weights,
                muons=muons,
                year=self._year,
                mod=self._yearmod,
            )
        # selections
        self.selections = PackedSelection()
        self.selections.add("trigger_ele", trigger["ele"])
        self.selections.add("trigger_mu", trigger["mu"])
        self.selections.add("lumi", lumi_mask)
        self.selections.add("metfilters", metfilters)
        self.selections.add("deltaR", ak.any(mu_bjet_dr > 0.4, axis=1))
        self.selections.add("two_bjets", n_good_bjets >= 1)
        self.selections.add("one_electron", n_good_electrons == 1)
        self.selections.add("one_muon", n_good_muons == 1)

        # regions
        regions = {
            "ele": {
                "numerator": [
                    "trigger_ele",
                    "trigger_mu",
                    "lumi",
                    "metfilters",
                    "two_bjets",
                    "one_muon",
                    "one_electron",
                ],
                "denominator": [
                    "trigger_mu",
                    "lumi",
                    "metfilters",
                    "two_bjets",
                    "one_muon",
                    "one_electron",
                ],
            },
            "mu": {
                "numerator": [
                    "trigger_ele",
                    "trigger_mu",
                    "lumi",
                    "metfilters",
                    "deltaR",
                    "two_bjets",
                    "one_electron",
                    "one_muon",
                ],
                "denominator": [
                    "trigger_ele",
                    "lumi",
                    "metfilters",
                    "deltaR",
                    "two_bjets",
                    "one_electron",
                    "one_muon",
                ],
            },
        }
        # weights per region
        common_weights = ["genweight", "L1Prefiring", "pileup", "btagSF"]
        electron_weights = ["electronReco", "electronID"]
        muon_weights = ["muonIso", "muonId"]

        numerator_weights = (
            common_weights
            + electron_weights
            + muon_weights
            + ["electronTrigger", "muonTriggerIso"]
        )
        denominator_weights = common_weights + electron_weights + muon_weights

        weights_per_region = {
            "ele": {
                "numerator": numerator_weights,
                "denominator": denominator_weights + ["muonTriggerIso"],
            },
            "mu": {
                "numerator": numerator_weights,
                "denominator": denominator_weights + ["electronTrigger"],
            },
        }

        # filling histograms
        def fill(region: str):
            selections = regions[self._channel][region]
            cut = self.selections.all(*selections)

            region_weights = weights_per_region[self._channel][region]
            region_weight = weights.partial_weight(region_weights)[cut]

            self.output["jet_kin"].fill(
                region=region,
                jet_pt=normalize(candidatebjet.pt, cut),
                jet_eta=normalize(candidatebjet.eta, cut),
                weight=region_weight,
            )
            self.output["met_kin"].fill(
                region=region,
                met_pt=normalize(met.pt, cut),
                met_phi=normalize(met.phi, cut),
                weight=region_weight,
            )

            self.output["electron_kin"].fill(
                region=region,
                electron_pt=normalize(electrons.pt, cut),
                electron_relIso=normalize(ele_reliso, cut),
                electron_eta=normalize(electrons.eta, cut),
                weight=region_weight,
            )
            self.output["muon_kin"].fill(
                region=region,
                muon_pt=normalize(muons.pt, cut),
                muon_relIso=normalize(mu_reliso, cut),
                muon_eta=normalize(muons.eta, cut),
                weight=region_weight,
            )
            self.output["electron_bjet_kin"].fill(
                region=region,
                electron_bjet_dr=normalize(ele_bjet_dr, cut),
                invariant_mass=normalize(ele_bjet_mass, cut),
                weight=region_weight,
            )
            self.output["muon_bjet_kin"].fill(
                region=region,
                muon_bjet_dr=normalize(mu_bjet_dr, cut),
                invariant_mass=normalize(mu_bjet_mass, cut),
                weight=region_weight,
            )
            self.output["lep_met_kin"].fill(
                region=region,
                electron_met_transverse_mass=normalize(ele_met_tranverse_mass, cut),
                muon_met_transverse_mass=normalize(mu_met_transverse_mass, cut),
                weight=region_weight,
            )
            self.output["lep_bjet_met_kin"].fill(
                region=region,
                electron_total_transverse_mass=normalize(
                    ele_total_transverse_mass, cut
                ),
                muon_total_transverse_mass=normalize(mu_total_transverse_mass, cut),
                weight=region_weight,
            )

            # cutflow
            cutflow_selections = []
            for selection in regions[self._channel][region]:
                cutflow_selections.append(selection)
                cutflow_cut = self.selections.all(*cutflow_selections)
                if self.isMC:
                    cutflow_weight = weights.partial_weight(region_weights)
                    self.output["cutflow"][region][selection] = np.sum(
                        cutflow_weight[cutflow_cut]
                    )
                else:
                    self.output["cutflow"][region][selection] = np.sum(cutflow_cut)

        for region in regions[self._channel]:
            fill(region)
        return {dataset: self.output}

    def postprocess(self, accumulator):
        return accumulator
