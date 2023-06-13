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
            
        # define the output accumulator
        self.make_output = lambda: {
            "sumw": 0,
            "cutflow": {},
            "weighted_cutflow": {},
            "jet_kin": hist.Hist(
                hist.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="jet_pt",
                    label=r"bJet $p_T$ [GeV]",
                ),
                hist.axis.Regular(50, -2.4, 2.4, name="jet_eta", label="bJet $\eta$"),
                hist.axis.Regular(50, -4.0, 4.0, name="jet_phi"),
                hist.storage.Weight(),
            ),
            "met_kin": hist.Hist(
                hist.axis.Variable(
                    [50, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="met",
                    label=r"$p_T^{miss}$ [GeV]",
                ),
                hist.axis.Regular(
                    50, -4.0, 4.0, name="met_phi", label=r"$\phi(p_T^{miss})$"
                ),
                hist.storage.Weight(),
            ),
            "lepton_kin": hist.Hist(
                hist.axis.Variable(
                    [30, 60, 90, 120, 150, 180, 210, 240, 300, 500],
                    name="lepton_pt",
                    label=r"lepton $p_T$ [GeV]",
                ),
                hist.axis.Regular(
                    25, 0, 1, name="lepton_relIso", label="lepton RelIso"
                ),
                hist.axis.Regular(
                    50, -2.4, 2.4, name="lepton_eta", label="lepton $\eta$"
                ),
                hist.axis.Regular(50, -4.0, 4.0, name="lepton_phi", label="lepton phi"),
                hist.storage.Weight(),
            ),
            "lepton_bjet_kin": hist.Hist(
                hist.axis.Regular(
                    30, 0, 5, name="lepton_bjet_dr", label="$\Delta R(\mu, bJet)$"
                ),
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500],
                    name="invariant_mass",
                    label=r"$m(\mu, bJet)$ [GeV]",
                ),
                hist.storage.Weight(),
            ),
            "lepton_met_kin": hist.Hist(
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="lepton_met_tranverse_mass",
                    label=r"$m_T(\mu, p_T^{miss})$ [GeV]",
                ),
                hist.axis.Regular(
                    30,
                    0,
                    4,
                    name="lepton_met_delta_phi",
                    label=r"$\Delta phi(\mu, p_T^{miss})$",
                ),
                hist.storage.Weight(),
            ),
            "lepton_bjet_met_kin": hist.Hist(
                hist.axis.Variable(
                    [40, 75, 100, 125, 150, 175, 200, 300, 500, 800],
                    name="lepton_total_transverse_mass",
                    label=r"$m_T^{tot}(\mu, bJet, p_T^{miss})$ [GeV]",
                ),
                hist.storage.Weight(),
            ),
        }

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        # get dataset name from metadata
        dataset = events.metadata["dataset"]
        
        # set output accumulator
        self.output = self.make_output()

        # save number of events
        nevents = len(events)
        self.output["cutflow"]["nevents"] = nevents

        # check if sample is MC
        self.is_mc = hasattr(events, "genWeight")

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
            (events.Muon.pt >= 30)
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
        leading_bjet = ak.firsts(bjets)

        # lepton relative isolation
        lepton_reliso = (
            leptons.pfRelIso04_all
            if hasattr(leptons, "pfRelIso04_all")
            else leptons.pfRelIso03_all
        )
        # lepton-bjet deltaR and invariant mass
        lepton_bjet_dr = leading_bjet.delta_r(leptons)
        lepton_bjet_mass = (leptons + leading_bjet).mass

        # lepton-MET transverse mass and deltaPhi
        lepton_met_tranverse_mass = np.sqrt(
            2.0
            * leptons.pt
            * met.pt
            * (ak.ones_like(met.pt) - np.cos(leptons.delta_phi(met)))
        )
        lepton_met_delta_phi = np.abs(leptons.delta_phi(met))
        
        # lepton-bJet-MET total transverse mass
        lepton_total_transverse_mass = np.sqrt(
            (leptons.pt + leading_bjet.pt + met.pt) ** 2
            - (leptons + leading_bjet + met).pt ** 2
        )
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

        # cross cleaning: check that bjets does not overlap with our selected leptons
        self.selections.add(
            "lepton_bjet_dr", ak.all(bjets.delta_r(leptons) > 0.4, axis=1)
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
                "lepton_bjet_dr",
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
                "lepton_bjet_dr",
            ],
        }
        # check how many events pass each selection
        cutflow_selections = []
        for selection in regions[self._channel]:
            cutflow_selections.append(selection)
            cutflow_cut = self.selections.all(*cutflow_selections)
            self.output["cutflow"][selection] = np.sum(cutflow_cut)
            if self.is_mc:
                self.output["weighted_cutflow"][selection] = np.sum(
                    events.genWeight[cutflow_cut]
                )
        # -------------
        # event weights
        # -------------
        # define weights container
        self.weights = Weights(nevents, storeIndividual=True)
        if self.is_mc:
            # add gen weigths
            gen_weight = events.genWeight
            self.weights.add("genweight", gen_weight)
            self.output["sumw"] = ak.sum(gen_weight)
            
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
        # get weight statistics
        weight_statistics = self.weights.weightStatistics

        # -----------------
        # histogram filling
        # -----------------
        # combine all region selections into a single mask
        selections = regions[self._channel]
        cut = self.selections.all(*selections)

        # get total weight from the weights container
        region_weight = self.weights.weight()[cut]

        # fill histograms
        self.output["jet_kin"].fill(
            jet_pt=normalize(leading_bjet.pt, cut),
            jet_eta=normalize(leading_bjet.eta, cut),
            jet_phi=normalize(leading_bjet.phi, cut),
            weight=region_weight,
        )
        self.output["met_kin"].fill(
            met=normalize(met.pt, cut),
            met_phi=normalize(met.phi, cut),
            weight=region_weight,
        )
        self.output["lepton_kin"].fill(
            lepton_pt=normalize(leptons.pt, cut),
            lepton_relIso=normalize(lepton_reliso, cut),
            lepton_eta=normalize(leptons.eta, cut),
            lepton_phi=normalize(leptons.phi, cut),
            weight=region_weight,
        )
        self.output["lepton_bjet_kin"].fill(
            lepton_bjet_dr=normalize(lepton_bjet_dr, cut),
            invariant_mass=normalize(lepton_bjet_mass, cut),
            weight=region_weight,
        )
        self.output["lepton_met_kin"].fill(
            lepton_met_tranverse_mass=normalize(lepton_met_tranverse_mass, cut),
            lepton_met_delta_phi=normalize(lepton_met_delta_phi, cut),
            weight=region_weight,
        )
        self.output["lepton_bjet_met_kin"].fill(
            lepton_total_transverse_mass=normalize(lepton_total_transverse_mass, cut),
            weight=region_weight,
        )

        return {
            dataset: self.output,
            "weight_statistics": weight_statistics,
        }

    def postprocess(self, accumulator):
        return accumulator