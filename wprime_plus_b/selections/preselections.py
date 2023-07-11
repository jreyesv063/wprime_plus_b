import json
import awkward as ak
import numpy as np

def select_good_electrons(events, channel, lepton_flavor):
    if channel == "2b1l":
        good_electron_pt = 55 if lepton_flavor == "ele" else 30
    if channel == "1b1e1mu":
        good_electron_pt = 55 if lepton_flavor == "ele" else 30
        
    good_electrons = (
        (events.Electron.pt >= good_electron_pt)
        & (np.abs(events.Electron.eta) < 2.4)
        & (
            (np.abs(events.Electron.eta) < 1.44)
            | (np.abs(events.Electron.eta) > 1.57)
        )
        & (
            events.Electron.mvaFall17V2Iso_WP80
            if lepton_flavor == "ele"
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
    
    return electrons, n_good_electrons


def select_good_muons(events):
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
    
    return muons, n_good_muons


def select_good_taus(events):
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
    
    return taus, n_good_taus


def select_good_bjets(jets, year="2017", working_point="M"):
    # open and load btagDeepFlavB working point
    with open("wprime_plus_b/data/btagWPs.json", "r") as handle:
        btagDeepFlavB = json.load(handle)["deepJet"][year][working_point]
        
    good_bjets = (
        (jets.pt >= 20)
        & (jets.jetId == 6)
        & (jets.puId == 7)
        & (jets.btagDeepFlavB > btagDeepFlavB)
        & (np.abs(jets.eta) < 2.4)
    )
    n_good_bjets = ak.sum(good_bjets, axis=1)
    bjets = jets[good_bjets]
    
    return bjets, n_good_bjets