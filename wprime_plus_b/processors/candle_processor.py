import os
import json
import pickle
import correctionlib
import numpy as np
import pandas as pd
import awkward as ak
import hist as hist2
from datetime import datetime
from typing import List, Union
from typing import Type
from coffea import util
from coffea import processor
from coffea.nanoevents.methods import candidate, vector
from coffea.analysis_tools import Weights, PackedSelection
from .utils import normalize, build_p4
from .corrections import (
    add_pileup_weight,
    add_electronID_weight,
    add_electronReco_weight,
    add_electronTrigger_weight,
    add_muon_weight,
    add_muonTriggerIso_weight,
    get_met_corrections,
)


class CandleProcessor(processor.ProcessorABC):
    def __init__(
        self,
        year: str = "2017",
        yearmod: str = "",
        channel: str = "ele",
        output_location: str = "",
        dir_name: str = "",
    ):
        self._year = year
        self._yearmod = yearmod
        self._channel = channel
        self._output_location = output_location
        self._dir_name = dir_name

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
        # output histograms
        self.make_output = lambda: {
            "sumw": 0,
            "mass": hist2.Hist(
                hist2.axis.Regular(
                    40, 50, 200, name="invariant_mass", label="$m_{ll}$ [GeV]"
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
                ak.pad_none(events.Electron, 2).charge[:, 0]
                * ak.pad_none(events.Electron, 2).charge[:, 1]
                < 0
            )
            & events.Electron.mvaFall17V2Iso_WP80
        )
        n_good_electrons = ak.sum(good_electrons, axis=1)
        electrons = events.Electron[good_electrons]

        leading_electron = build_p4(ak.pad_none(electrons, 2)[:, 0])
        subleading_electron = build_p4(ak.pad_none(electrons, 2)[:, 1])
        # muons
        good_muons = (
            (events.Muon.pt >= 35)
            & (np.abs(events.Muon.eta) < 2.4)
            & (
                ak.pad_none(events.Muon, 2).charge[:, 0]
                * ak.pad_none(events.Muon, 2).charge[:, 1]
                < 0
            )
            & (events.Muon.mediumId)
        )
        n_good_muons = ak.sum(good_muons, axis=1)
        muons = events.Muon[good_muons]

        leading_muon = build_p4(ak.pad_none(muons, 2)[:, 0])
        subleading_muon = build_p4(ak.pad_none(muons, 2)[:, 1])

        # invariant mass
        invariant_mass = {
            "ele": (leading_electron + subleading_electron).mass,
            "mu": (leading_muon + subleading_muon).mass,
        }
        # weights
        self.weights = Weights(nevents, storeIndividual=True)
        if self.isMC:
            # genweight
            self.output["sumw"] = ak.sum(events.genWeight)
            self.weights.add("genweight", events.genWeight)
            # L1prefiring
            if self._year in ("2016", "2017"):
                self.weights.add(
                    "L1Prefiring",
                    weight=events.L1PreFiringWeight.Nom,
                    weightUp=events.L1PreFiringWeight.Up,
                    weightDown=events.L1PreFiringWeight.Dn,
                )
            # pileup
            add_pileup_weight(
                weights=self.weights,
                year=self._year,
                mod=self._yearmod,
                nPU=ak.to_numpy(events.Pileup.nPU),
            )
            # electron weights
            if self._channel == "ele":
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
                add_electronTrigger_weight(
                    weights=self.weights,
                    electrons=electrons,
                    year=self._year,
                    mod=self._yearmod,
                )
            # muon weights
            if self._channel == "mu":
                add_muon_weight(
                    weights=self.weights,
                    muons=muons,
                    sf_type="id",
                    year=self._year,
                    mod=self._yearmod,
                    wp="tight",
                )
                add_muon_weight(
                    weights=self.weights,
                    muons=muons,
                    sf_type="iso",
                    year=self._year,
                    mod=self._yearmod,
                    wp="tight",
                )
                add_muonTriggerIso_weight(
                    weights=self.weights,
                    muons=muons,
                    year=self._year,
                    mod=self._yearmod,
                )
        # selections
        n_good_leptons = {"ele": n_good_electrons == 2, "mu": n_good_muons == 2}
        self.selections = PackedSelection()
        self.selections.add("trigger_ele", trigger["ele"])
        self.selections.add("trigger_mu", trigger["mu"])
        self.selections.add("lumi", lumi_mask)
        self.selections.add("metfilters", metfilters)
        self.selections.add("two_leptons", n_good_leptons[self._channel])
        self.selections.add(
            "mass_range",
            (60 < invariant_mass[self._channel])
            & (invariant_mass[self._channel] < 120),
        )

        # regions
        regions = {
            "ele": [
                "lumi",
                "metfilters",
                "trigger_ele",
                "mass_range",
                "two_leptons",
            ],
            "mu": [
                "lumi",
                "metfilters",
                "trigger_mu",
                "mass_range",
                "two_leptons",
            ],
        }

        selections = regions[self._channel]
        cut = self.selections.all(*selections)
        region_weight = self.weights.weight()[cut]

        self.output["mass"].fill(
            invariant_mass=normalize(invariant_mass[self._channel], cut),
            weight=region_weight,
        )

        return {dataset: self.output}

    def postprocess(self, accumulator):
        return accumulator
