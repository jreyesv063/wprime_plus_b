import json
import copy
import pickle
import numpy as np
import awkward as ak
from coffea import processor
from wprime_plus_b.processors import utils
from coffea.analysis_tools import Weights, PackedSelection
from wprime_plus_b.corrections.btag import BTagCorrector
from wprime_plus_b.corrections.jec import jet_corrections
from wprime_plus_b.corrections.met import met_phi_corrections
from wprime_plus_b.corrections.pileup import add_pileup_weight
from wprime_plus_b.corrections.lepton import ElectronCorrector, MuonCorrector
from wprime_plus_b.selections.preselections import (
    select_good_electrons,
    select_good_muons,
    select_good_taus,
    select_good_bjets,
)



class TtbarAnalysis(processor.ProcessorABC):
    """
    Ttbar Analysis processor

    Parameters:
    -----------
    channel:
        region channel {'2b1l', '1b1e1mu'}
    lepton_flavor:
        lepton flavor {'ele', 'mu'}
    year:
        year of the dataset {"2016", "2017", "2018"}
    year_mode:
        year modifier {"", "APV"}
    btag_wp:
        working point of the deepJet tagger
    """

    def __init__(
        self,
        channel: str = "2b1l",
        lepton_flavor: str = "ele",
        year: str = "2017",
        yearmod: str = "",
        btag_wp: str = "M",
        syst: str = "nominal",
    ):
        self._year = year
        self._yearmod = yearmod
        self._lepton_flavor = lepton_flavor
        self._channel = channel
        self.btag_wp = btag_wp
        self.syst = syst

        # define regions of the analysis
        channels = ["2b1l", "1b1e1mu"]
        lepton_flavors = ["ele", "mu"]
        self.regions = [
            f"{ch}_{lep}" for ch in channels for lep in lepton_flavors
        ]  # ["2b1l_ele", "2b1l_mu", "1b1e1mu_ele", "1b1e1mu_mu"]

        # initialize dictionary of hists for control regions
        self.hist_dict = {}
        for region in self.regions:
            self.hist_dict[region] = {
                "jet_kin": utils.histograms.jet_hist,
                "met_kin": utils.histograms.met_hist,
                "lepton_kin": utils.histograms.lepton_hist,
                "lepton_bjet_kin": utils.histograms.lepton_bjet_hist,
                "lepton_met_kin": utils.histograms.lepton_met_hist,
                "lepton_met_bjet_kin": utils.histograms.lepton_met_bjet_hist,
            }
        # define dictionary to store analysis variables
        self.features = {}

    def add_feature(self, name: str, var: ak.Array) -> None:
        """add a variable array to the out dictionary"""
        self.features = {**self.features, name: var}

    def process(self, events):
        # get dataset name
        dataset = events.metadata["dataset"]

        # get number of events before selection
        nevents = len(events)

        # check if sample is MC
        self.is_mc = hasattr(events, "genWeight")

        # create copies of histogram objects
        hist_dict = copy.deepcopy(self.hist_dict)

        syst_variations = ["nominal"]
        for syst_var in syst_variations:
            # ------------------
            # event preselection
            # ------------------
            # select good electrons
            electrons, n_good_electrons = select_good_electrons(
                events, channel=self._channel, lepton_flavor=self._lepton_flavor
            )

            # select good muons
            muons, n_good_muons = select_good_muons(events)

            # select good taus
            taus, n_good_taus = select_good_taus(events)

            # apply JEC/JER corrections to MC jets (propagate corrections to MET)
            # in data, the corrections are already applied
            if self.is_mc:
                jets, met = jet_corrections(events, self._year + self._yearmod)
            else:
                jets, met = events.Jet, events.MET
            
            # select good bjets
            bjets, n_good_bjets = select_good_bjets(
                jets=jets, year=self._year, working_point=self.btag_wp
            )

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
            # event selection
            # ---------------
            # make a PackedSelection object to store selection masks
            self.selections = PackedSelection()

            # add luminosity calibration mask (only to data)
            # open and load lumi masks
            with open("wprime_plus_b/data/lumi_masks.pkl", "rb") as handle:
                self._lumi_mask = pickle.load(handle)
            if not self.is_mc:
                lumi_mask = self._lumi_mask[self._year](
                    events.run, events.luminosityBlock
                )
            else:
                lumi_mask = np.ones(len(events), dtype="bool")
            self.selections.add("lumi", lumi_mask)

            # add lepton triggers masks
            with open("wprime_plus_b/data/triggers.json", "r") as handle:
                self._triggers = json.load(handle)[self._year]
            trigger = {}
            for ch in ["ele", "mu"]:
                trigger[ch] = np.zeros(nevents, dtype="bool")
                for t in self._triggers[ch]:
                    if t in events.HLT.fields:
                        trigger[ch] = trigger[ch] | events.HLT[t]
            self.selections.add("trigger_ele", trigger["ele"])
            self.selections.add("trigger_mu", trigger["mu"])

            # add MET filters mask
            # open and load met filters
            with open("wprime_plus_b/data/metfilters.json", "rb") as handle:
                self._metfilters = json.load(handle)[self._year]
            metfilters = np.ones(nevents, dtype="bool")
            metfilterkey = "mc" if self.is_mc else "data"
            for mf in self._metfilters[metfilterkey]:
                if mf in events.Flag.fields:
                    metfilters = metfilters & events.Flag[mf]
            self.selections.add("metfilters", metfilters)

            # check that there be a minimum MET greater than 50 GeV
            self.selections.add("met_pt", met.pt > 50)

            # cross-cleaning
            # We can define the variables for leptons from just the leading (in pt) lepton
            # since all of our signal and control regions require exactly zero or one of
            # them so there is no ambiguity to resolve.
            # Some control regions require more than one bjet though
            leading_bjets = ak.firsts(bjets)
            subleading_bjets = ak.pad_none(bjets, 2)[:, 1]

            # common
            # check that leading bjets does not overlap with our selected leptons
            self.selections.add(
                "electron_leadingbjet_dr",
                leading_bjets.delta_r(ak.firsts(electrons)) > 0.4,
            )
            self.selections.add(
                "muon_leadingbjet_dr", leading_bjets.delta_r(ak.firsts(muons)) > 0.4
            )

            # 2b1l region
            # check that subleading bjets does not overlap with our selected leptons
            self.selections.add(
                "muon_subleadingbjet_dr",
                subleading_bjets.delta_r(ak.firsts(muons)) > 0.4,
            )
            self.selections.add(
                "electron_subleadingbjet_dr",
                subleading_bjets.delta_r(ak.firsts(electrons)) > 0.4,
            )

            # 1b1e1muregion
            # check that selected leptons does not overlap
            self.selections.add(
                "electron_muon_dr", ak.firsts(electrons).delta_r(ak.firsts(muons)) > 0.4
            )

            # add number of leptons and bjets
            # common
            self.selections.add("one_electron", n_good_electrons == 1)
            self.selections.add("one_muon", n_good_muons == 1)
            self.selections.add("tau_veto", n_good_taus == 0)
            # 2b1l region
            self.selections.add("two_bjets", n_good_bjets == 2)
            self.selections.add("electron_veto", n_good_electrons == 0)
            self.selections.add("muon_veto", n_good_muons == 0)
            # 1b1e1mu region
            self.selections.add("one_bjet", n_good_bjets == 1)

            # define selection regions for each channel
            region_selection = {
                "2b1l": {
                    "ele": [
                        "lumi",
                        "metfilters",
                        "trigger_ele",
                        "met_pt",
                        "two_bjets",
                        "tau_veto",
                        "muon_veto",
                        "one_electron",
                        "electron_leadingbjet_dr",
                        "electron_subleadingbjet_dr",
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
                        "muon_leadingbjet_dr",
                        "muon_subleadingbjet_dr",
                    ],
                },
                "1b1e1mu": {
                    "ele": [
                        "lumi",
                        "metfilters",
                        "trigger_mu",
                        "met_pt",
                        "one_bjet",
                        "tau_veto",
                        "one_muon",
                        "one_electron",
                        "muon_leadingbjet_dr",
                        "electron_leadingbjet_dr",
                        "electron_muon_dr",
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
                        "muon_leadingbjet_dr",
                        "electron_leadingbjet_dr",
                        "electron_muon_dr",
                    ],
                },
            }

            self.selections.add(
                "2b1l_ele", self.selections.all(*region_selection["2b1l"]["ele"])
            )
            self.selections.add(
                "2b1l_mu", self.selections.all(*region_selection["2b1l"]["mu"])
            )
            self.selections.add(
                "1b1e1mu_ele", self.selections.all(*region_selection["1b1e1mu"]["ele"])
            )
            self.selections.add(
                "1b1e1mu_mu", self.selections.all(*region_selection["1b1e1mu"]["mu"])
            )

            for region in self.regions:
                if not f"{self._channel}_{self._lepton_flavor}" in region:
                    continue
                # ---------------
                # event variables
                # ---------------
                region_selection = self.selections.all(region)

                # if there are no events left after selection cuts continue to the next .root file
                if ak.sum(region_selection) == 0:
                    continue

                # select region objects
                region_bjets = bjets[region_selection]
                region_electrons = electrons[region_selection]
                region_muons = muons[region_selection]
                region_met = met[region_selection]

                # define region leptons
                region_leptons = (
                    region_electrons if self._lepton_flavor == "ele" else region_muons
                )
                # lepton relative isolation
                lepton_reliso = (
                    region_leptons.pfRelIso04_all
                    if hasattr(region_leptons, "pfRelIso04_all")
                    else region_leptons.pfRelIso03_all
                )
                # leading bjets
                leading_bjets = ak.firsts(region_bjets)

                # lepton-bjet deltaR and invariant mass
                lepton_bjet_dr = leading_bjets.delta_r(region_leptons)
                lepton_bjet_mass = (region_leptons + leading_bjets).mass

                # lepton-MET transverse mass and deltaPhi
                lepton_met_mass = np.sqrt(
                    2.0
                    * region_leptons.pt
                    * region_met.pt
                    * (
                        ak.ones_like(region_met.pt)
                        - np.cos(region_leptons.delta_phi(region_met))
                    )
                )
                lepton_met_delta_phi = np.abs(region_leptons.delta_phi(region_met))

                # lepton-bJet-MET total transverse mass
                lepton_met_bjet_mass = np.sqrt(
                    (region_leptons.pt + leading_bjets.pt + region_met.pt) ** 2
                    - (region_leptons + leading_bjets + region_met).pt ** 2
                )

                self.add_feature("lepton_pt", region_leptons.pt)
                self.add_feature("lepton_eta", region_leptons.eta)
                self.add_feature("lepton_phi", region_leptons.phi)
                self.add_feature("jet_pt", leading_bjets.pt)
                self.add_feature("jet_eta", leading_bjets.eta)
                self.add_feature("jet_phi", leading_bjets.phi)
                self.add_feature("met", region_met.pt)
                self.add_feature("met_phi", region_met.phi)
                self.add_feature("lepton_bjet_dr", lepton_bjet_dr)
                self.add_feature("lepton_bjet_mass", lepton_bjet_mass)
                self.add_feature("lepton_met_mass", lepton_met_mass)
                self.add_feature("lepton_met_delta_phi", lepton_met_delta_phi)
                self.add_feature("lepton_met_bjet_mass", lepton_met_bjet_mass)

                # -------------
                # event weights
                # -------------
                weights_container = Weights(
                    len(events[region_selection]), storeIndividual=True
                )
                if self.is_mc:
                    # add gen weigths
                    gen_weight = events.genWeight[region_selection]
                    weights_container.add("genweight", gen_weight)

                    # add L1prefiring weights
                    if self._year in ("2016", "2017"):
                        weights_container.add(
                            "L1Prefiring",
                            weight=events.L1PreFiringWeight.Nom[region_selection],
                            weightUp=events.L1PreFiringWeight.Up[region_selection],
                            weightDown=events.L1PreFiringWeight.Dn[region_selection],
                        )
                    # add pileup reweighting
                    add_pileup_weight(
                        n_true_interactions=ak.to_numpy(
                            events.Pileup.nPU[region_selection]
                        ),
                        weights=weights_container,
                        year=self._year,
                        year_mod=self._yearmod,
                    )
                    # b-tagging corrector
                    njets = 1 if self._channel == "1b1e1mu" else 2
                    btag_corrector = BTagCorrector(
                        jets=region_bjets,
                        njets=njets,
                        weights=weights_container,
                        sf_type="comb",
                        worging_point="M",
                        tagger="deepJet",
                        year=self._year,
                        year_mod=self._yearmod,
                        full_run=False,
                    )
                    # add b-tagging weights
                    btag_corrector.add_btag_weights(flavor="bc")
                    # if self._channel == "2b1l":
                    #    btag_corrector.add_btag_weights(flavor="light")

                    if self._channel == "1b1e1mu" or (self._channel == "2b1l" and self._lepton_flavor == "ele"):
                        # electron corrector
                        electron_corrector = ElectronCorrector(
                            electrons=ak.firsts(region_electrons),
                            weights=weights_container,
                            year=self._year,
                            year_mod=self._yearmod,
                            tag="leading_electron",
                        )
                        # add electron ID weights
                        electron_corrector.add_id_weight(
                            working_point="wp80noiso"
                            if self._lepton_flavor == "ele"
                            else "wp90noiso",
                        )
                        # add electron reco weights
                        electron_corrector.add_reco_weight()
                    
                    if self._channel == "1b1e1mu" or (self._channel == "2b1l" and self._lepton_flavor == "mu"):
                        # muon corrector
                        muon_corrector = MuonCorrector(
                            muons=ak.firsts(region_muons),
                            weights=weights_container,
                            year=self._year,
                            year_mod=self._yearmod,
                            tag="leading_muon",
                        )
                        # add muon ID weights
                        muon_corrector.add_id_weight(working_point="tight")

                        # add muon iso weights
                        muon_corrector.add_iso_weight(working_point="tight")

                    # add trigger weights
                    if self._channel == "1b1e1mu":
                        if self._lepton_flavor == "ele":
                            muon_corrector.add_triggeriso_weight()
                        else:
                            electron_corrector.add_trigger_weight()
                    if self._channel == "2b1l":
                        if self._lepton_flavor == "ele":
                            electron_corrector.add_trigger_weight()
                        else:
                            muon_corrector.add_triggeriso_weight()
                            
                # get total weight from the weights container
                region_weights = weights_container.weight()
                
                # -----------------------------
                # fill histograms
                # -----------------------------
                syst_var_name = f"{syst_var}"
                for kin in hist_dict[region]:
                    fill_args = {
                        feature: utils.analysis_utils.normalize(self.features[feature])
                        for feature in hist_dict[region][kin].axes.name[:-1]
                        if "dataset" not in feature
                    }
                    hist_dict[region][kin].fill(
                        **fill_args,
                        dataset=dataset,
                        variation=syst_var_name,
                        weight=region_weights,
                    )
        # define output
        output = {
            "histograms": hist_dict[f"{self._channel}_{self._lepton_flavor}"],
            "metadata": {
                "events_before": nevents,
                "events_after": ak.sum(region_selection),
                "filenames": f"{events.metadata['filename']}\n",
            },
        }
        # if dataset is montecarlo add sumw to output
        if self.is_mc:
            output.update({"sumw": ak.sum(events.genWeight)})
        return output

    def postprocess(self, accumulator):
        return accumulator