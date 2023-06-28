import json
import gzip
import cloudpickle
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from .utils import clip_array
from typing import Type, Tuple
from coffea.lookup_tools import extractor
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory


# CorrectionLib files are available from
POG_CORRECTION_PATH = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration"

# summary of pog scale factors: https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/
POG_JSONS = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "btag": ["BTV", "btagging.json.gz"],
    "met": ["JME", "met.json.gz"],
}

POG_YEARS = {
    "2016": "2016postVFP_UL",
    "2016APV": "2016preVFP_UL",
    "2017": "2017_UL",
    "2018": "2018_UL",
}

TAGGER_BRANCH = {
    "deepJet": "btagDeepFlavB",
    "deepCSV": "btagDeep",
}


def get_pog_json(json_name: str, year: str) -> str:
    """
    returns the path to the pog json file

    Parameters:
    -----------
        json_name:
            json name {'muon', 'electron', 'pileup', 'btag'}
        year:
            dataset year {'2016', '2017', '2018'}
    """
    if json_name in POG_JSONS:
        pog_json = POG_JSONS[json_name]
    else:
        print(f"No json for {json_name}")
    return f"{POG_CORRECTION_PATH}/POG/{pog_json[0]}/{POG_YEARS[year]}/{pog_json[1]}"


# ------------------------------
# EVENT-WISE CORRECTIONS
# ------------------------------


# --------------------
# pileup scale factors
# --------------------
def add_pileup_weight(
    weights: Type[Weights], year: str, year_mod: str, n_true_interactions: ak.Array
) -> None:
    """
    add pileup scale factor

    Parameters:
    -----------
        n_true_interactions:
            number of true interactions (events.Pileup.nPU)
        weights:
            Weights object from coffea.analysis_tools
        year:
            dataset year {'2016', '2017', '2018'}
        year_mod:
            year modifier {"", "APV"}
    """
    # define correction set
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="pileup", year=year + year_mod)
    )
    # define goldenJSON file names
    year_to_corr = {
        "2016": "Collisions16_UltraLegacy_goldenJSON",
        "2017": "Collisions17_UltraLegacy_goldenJSON",
        "2018": "Collisions18_UltraLegacy_goldenJSON",
    }
    # get scale factors
    values = {}
    values["nominal"] = cset[year_to_corr[year]].evaluate(
        n_true_interactions, "nominal"
    )
    values["up"] = cset[year_to_corr[year]].evaluate(n_true_interactions, "up")
    values["down"] = cset[year_to_corr[year]].evaluate(n_true_interactions, "down")

    # add pileup scale factors to weights container
    weights.add(
        name="pileup",
        weight=values["nominal"],
        weightUp=values["up"],
        weightDown=values["down"],
    )


class BTagCorrector:
    """
        BTag corrector class.

        Parameters:
        -----------
            sf_type:
                scale factors type to use {mujets, comb}
                For the working point corrections the SFs in 'mujets' and 'comb' are for b/c jets.
                The 'mujets' SFs contain only corrections derived in QCD-enriched regions.
                The 'comb' SFs contain corrections derived in QCD and ttbar-enriched regions.
                Hence, 'comb' SFs can be used everywhere, except for ttbar-dileptonic enriched analysis regions.
                For the ttbar-dileptonic regionsthe 'mujets' SFs should be used.
            worging_point:
                worging point {'L', 'M', 'T'}
            tagger:
                tagger {'deepJet', 'deepCSV'}
            year:
                dataset year {'2016', '2017', '2018'}
            year_mod:
                year modifier {"", "APV"}
            jets:
                Jet collection
            njets:
                Number of jets to use
            weights:
                Weights container from coffea.analysis_tools
            full_run:
                False (default) if only one year is analized,
                True if the fullRunII data is analyzed.
                If False, the 'up' and 'down' systematics are be used.
                If True, 'up/down_correlated' and 'up/down_uncorrelated'
                systematics are used instead of the 'up/down' ones,
                which are supposed to be correlated/decorrelated
                between the different data years

        Example:
        --------
            # load events array
            events = NanoEventsFactory.from_root('nanoaod_file.root', schemaclass=NanoAODSchema).events()

            # define your jet selection
            bjets = events.Jet[(
                (events.Jet.pt >= 20)
                & (events.Jet.jetId == 6)
                & (events.Jet.puId == 7)
                & (events.Jet.btagDeepFlavB > 0.3)
                & (np.abs(events.Jet.eta) < 2.4)
            )]

            # create an instance of the Weights container
            weights = Weights(len(events), storeIndividual=True)

            # create an instance of BTagCorrector
            btag_corrector = BTagCorrector(
                jets=bjets,
                njets=2,
                weights=weights,
                sf_type="comb",
                worging_point="M",
                tagger="deepJet",
                year="2017",
            )
            # add bc and light btagging weights to weights container
            btag_corrector.add_btag_weights(flavor="bc")
            btag_corrector.add_btag_weights(flavor="light")
    """

    def __init__(
        self,
        jets: ak.Array,
        njets: int,
        weights: Type[Weights],
        sf_type: str = "comb",
        worging_point: str = "M",
        tagger: str = "deepJet",
        year: str = "2017",
        year_mod: str = "",
        full_run: bool = False,
    ) -> None:
        self._sf = sf_type
        self._year = year
        self._yearmod = year_mod
        self._tagger = tagger
        self._wp = worging_point
        self._branch = TAGGER_BRANCH[tagger]
        self._weights = weights
        self._full_run = full_run

        # define correction set
        self._cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="btag", year=year + year_mod)
        )
        # systematics
        self._syst_up = "up_correlated" if full_run else "up"
        self._syst_down = "down_correlated" if full_run else "down"

        # bc and light jets
        # hadron flavor definition: 5=b, 4=c, 0=udsg
        self._bc_jets = jets[jets.hadronFlavour > 0]
        self._light_jets = jets[jets.hadronFlavour == 0]
        self._jet_map = {"bc": self._bc_jets, "light": self._light_jets}

        # number of jets to use
        if njets == "all":
            njets = ak.max(ak.num(jets))
        self._njets = njets

        # load efficiency lookup table (only for deepJet)
        # efflookup(pt, |eta|, flavor)
        with importlib.resources.path(
            "wprime_plus_b.data", f"btag_eff_{self._tagger}_{self._wp}_{year}.coffea"
        ) as filename:
            self._efflookup = util.load(str(filename))
            
        # load btagging working point (only for deepJet)
        # https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        with importlib.resources.path("wprime_plus_b.data", "btagWPs.json") as path:
            with open(path, "r") as handle:
                btag_working_points = json.load(handle)
        self._btagwp = btag_working_points[tagger][year + year_mod][worging_point]

    def add_btag_weights(self, flavor: str) -> None:
        """
        Add b-tagging weights (nominal, up and down) to weights container for bc or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {'bc', 'light'}
        """
        # efficiencies
        eff = self.efficiency(flavor=flavor)

        # mask with events that pass the btag working point
        passbtag = self.passbtag_mask(flavor=flavor)

        # scale factors
        jets_sf = self.get_scale_factors(flavor=flavor, syst="central")
        jets_sf_up = self.get_scale_factors(flavor=flavor, syst=self._syst_up)
        jets_sf_down = self.get_scale_factors(flavor=flavor, syst=self._syst_down)

        # get weights
        jets_weight = self.get_btag_weight(eff, jets_sf, passbtag)
        jets_weight_up = self.get_btag_weight(eff, jets_sf_up, passbtag)
        jets_weight_down = self.get_btag_weight(eff, jets_sf_down, passbtag)

        # add weights to Weights container
        self._weights.add(
            name=f"{flavor}_{self._njets}_jets",
            weight=jets_weight,
            weightUp=jets_weight_up,
            weightDown=jets_weight_down,
        )

    def efficiency(self, flavor: str, fill_value=1) -> ak.Array:
        """compute the btagging efficiency for 'njets' jets"""
        eff = self._efflookup(
            self._jet_map[flavor].pt,
            np.abs(self._jet_map[flavor].eta),
            self._jet_map[flavor].hadronFlavour,
        )
        return clip_array(
            array=eff,
            target=self._njets,
            fill_value=fill_value,
        )

    def passbtag_mask(self, flavor, fill_value=True) -> ak.Array:
        """return the mask with jets that pass the b-tagging working point"""
        pass_mask = self._jet_map[flavor][self._branch] > self._btagwp
        return clip_array(array=pass_mask, target=self._njets, fill_value=fill_value)

    def get_scale_factors(self, flavor: str, syst="central", fill_value=1) -> ak.Array:
        """
        compute jets scale factors
        """
        scale_factors = self.get_sf(flavor=flavor, syst=syst)
        return clip_array(
            array=scale_factors, target=self._njets, fill_value=fill_value
        )

    def get_sf(self, flavor: str, syst: str = "central") -> ak.Array:
        """
        compute the scale factors for bc or light jets

        Parameters:
        -----------
            flavor:
                hadron flavor {'bc', 'light'}
            syst:
                Name of the systematic {'central', 'down', 'down_correlated', 'down_uncorrelated', 'up', 'up_correlated'}
        """
        cset_keys = {
            "bc": f"{self._tagger}_{self._sf}",
            "light": f"{self._tagger}_incl",
        }
        # until correctionlib handles jagged data natively we have to flatten and unflatten
        j, nj = ak.flatten(self._jet_map[flavor]), ak.num(self._jet_map[flavor])
        sf = self._cset[cset_keys[flavor]].evaluate(
            syst,
            self._wp,
            np.array(j.hadronFlavour),
            np.array(abs(j.eta)),
            np.array(j.pt),
        )
        return ak.unflatten(sf, nj)

    @staticmethod
    def get_btag_weight(eff: ak.Array, sf: ak.Array, passbtag: ak.Array) -> ak.Array:
        """
        compute b-tagging weights

        see: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BTagSFMethods

        Parameters:
        -----------
            eff:
                btagging efficiencies
            sf:
                jets scale factors
            passbtag:
                mask with jets that pass the b-tagging working point
        """
        # tagged SF = SF * eff / eff = SF
        tagged_sf = sf.mask[passbtag]

        # untagged SF = (1 - SF * eff) / (1 - eff)
        untagged_sf = ((1 - sf * eff) / (1 - eff)).mask[~passbtag]

        # if njets > 1, compute the product of the scale factors
        if tagged_sf.ndim > 1:
            tagged_sf = ak.prod(tagged_sf, axis=-1)
            untagged_sf = ak.prod(untagged_sf, axis=-1)
        return ak.fill_none(tagged_sf * untagged_sf, 1.0)


# ----------------------------------
# lepton scale factors
# -----------------------------------
#
# Electron
#    - ID: wp80noiso?
#    - Recon: RecoAbove20?
#    - Trigger: ?
#
# working points: (Loose, Medium, RecoAbove20, RecoBelow20, Tight, Veto, wp80iso, wp80noiso, wp90iso, wp90noiso)
#
class ElectronCorrector:
    """
    Electron corrector class

    Parameters:
    -----------
    electrons:
        electron collection
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset {'2016', '2017', '2018'}
    year_mod:
        Year modifier {'', 'APV'}
    tag:
        label to include in the weight name
    """

    def __init__(
        self,
        electrons: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        year_mod: str = "",
        tag: str = "electron",
    ) -> None:
        # electron array
        self.electrons = electrons

        # electron transverse momentum and pseudorapidity
        self.electron_pt = np.array(ak.fill_none(self.electrons.pt, 0.0))
        self.electron_eta = np.array(ak.fill_none(self.electrons.eta, 0.0))

        # weights container
        self.weights = weights

        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="electron", year=year + year_mod)
        )
        self.year = year
        self.year_mod = year_mod
        self.pog_year = POG_YEARS[year + year_mod]

        self.tag = tag

    def add_id_weight(self, working_point: str = "wp80noiso") -> None:
        """
        add electron identification scale factors to weights container

        Parameters:
        -----------
            working_point:
                Working point {'Loose', 'Medium', 'Tight', 'wp80iso', 'wp80noiso', 'wp90iso', 'wp90noiso'}
        """
        # electron pseudorapidity range: (-inf, inf)
        electron_eta = self.electron_eta

        # electron pt range: [10, inf)
        electron_pt = np.clip(
            self.electron_pt.copy(), 10.0, 499.999
        )  # potential problems with pt > 500 GeV

        # remove '_UL' from year
        year = self.pog_year.replace("_UL", "")

        # get scale factors
        values = {}
        values["nominal"] = self.cset["UL-Electron-ID-SF"].evaluate(
            year, "sf", working_point, electron_eta, electron_pt
        )
        values["up"] = self.cset["UL-Electron-ID-SF"].evaluate(
            year, "sfup", working_point, electron_eta, electron_pt
        )
        values["down"] = self.cset["UL-Electron-ID-SF"].evaluate(
            year, "sfdown", working_point, electron_eta, electron_pt
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"{self.tag}_id",
            weight=values["nominal"],
            weightUp=values["up"],
            weightDown=values["down"],
        )

    def add_reco_weight(self) -> None:
        """add electron reconstruction scale factors to weights container"""
        # electron pseudorapidity range: (-inf, inf)
        electron_eta = self.electron_eta

        # electron pt range: (20, inf)
        electron_pt = np.clip(
            self.electron_pt.copy(), 20.1, 499.999
        )  # potential problems with pt > 500 GeV

        # remove _UL from year
        year = self.pog_year.replace("_UL", "")

        # get scale factors
        values = {}
        values["nominal"] = self.cset["UL-Electron-ID-SF"].evaluate(
            year, "sf", "RecoAbove20", electron_eta, electron_pt
        )
        values["up"] = self.cset["UL-Electron-ID-SF"].evaluate(
            year, "sfup", "RecoAbove20", electron_eta, electron_pt
        )
        values["down"] = self.cset["UL-Electron-ID-SF"].evaluate(
            year, "sfdown", "RecoAbove20", electron_eta, electron_pt
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"{self.tag}_reco",
            weight=values["nominal"],
            weightUp=values["up"],
            weightDown=values["down"],
        )

    def add_trigger_weight(self) -> None:
        """add electron trigger scale factors to weights container"""
        # corrections still not provided by POG
        with importlib.resources.path(
            "wprime_plus_b.data", f"electron_trigger_{self.pog_year}.json"
        ) as filename:
            # correction set
            cset = correctionlib.CorrectionSet.from_file(str(filename))

            # electron pt range: (10, 500)
            electron_pt = np.clip(self.electron_pt.copy(), 10, 499.999)

            # electron pseudorapidity range: (-2.5, 2.5)
            electron_eta = np.clip(self.electron_eta.copy(), -2.499, 2.499)

            # get scale factors (only nominal)
            values = {}
            values["nominal"] = cset["UL-Electron-Trigger-SF"].evaluate(
                electron_eta, electron_pt
            )
            # add scale factors to weights container
            self.weights.add(
                name=f"{self.tag}_trigger",
                weight=values["nominal"],
            )


# Muon
#
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2016
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2017
# https://twiki.cern.ch/twiki/bin/view/CMS/MuonUL2018
#
#    - ID: medium prompt ID NUM_MediumPromptID_DEN_TrackerMuon?
#    - Iso: LooseRelIso with mediumID (NUM_LooseRelIso_DEN_MediumID)?
#    - Trigger iso:
#          2016: for IsoMu24 (and IsoTkMu24?) NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight?
#          2017: for isoMu27 NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight?
#          2018: for IsoMu24 NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight?
#
class MuonCorrector:
    """
    Muon corrector class

    Parameters:
    -----------
    muons:
        muons collection
    weights:
        Weights object from coffea.analysis_tools
    year:
        Year of the dataset {'2016', '2017', '2018'}
    year_mod:
        Year modifier {'', 'APV'}
    tag:
        label to include in the weight name
    """

    def __init__(
        self,
        muons: ak.Array,
        weights: Type[Weights],
        year: str = "2017",
        year_mod: str = "",
        tag: str = "muon",
    ) -> None:
        # muon array
        self.muons = muons

        # muon transverse momentum and pseudorapidity
        self.muon_pt = np.array(ak.fill_none(self.muons.pt, 0.0))
        self.muon_eta = np.array(ak.fill_none(self.muons.eta, 0.0))

        # weights container
        self.weights = weights

        # define correction set
        self.cset = correctionlib.CorrectionSet.from_file(
            get_pog_json(json_name="muon", year=year + year_mod)
        )

        self.year = year
        self.year_mod = year_mod
        self.pog_year = POG_YEARS[year + year_mod]

        self.tag = tag

    def add_id_weight(self, working_point: str = "tight") -> None:
        """
        add muon ID scale factors to weights container
        Parameters:
        -----------
            working_point:
                Working point {'medium', 'tight'}
        """
        self.add_weight(sf_type="id", working_point=working_point)

    def add_iso_weight(self, working_point: str = "tight") -> None:
        """
        add muon Iso (LooseRelIso with mediumID) scale factors to weights container
        Parameters:
        -----------
            working_point:
                Working point {'medium', 'tight'}
        """
        self.add_weight(sf_type="iso", working_point=working_point)

    def add_triggeriso_weight(self) -> None:
        """add muon Trigger Iso (IsoMu24 or IsoMu27) scale factors"""
        # muon absolute pseudorapidity range: [0, 2.4)
        muon_eta = np.clip(self.muon_eta.copy(), 0.0, 2.399)

        # muon pt range: [29, 200)
        muon_pt = np.clip(self.muon_pt.copy(), 29.0, 199.999)

        # scale factors keys
        sfs_keys = {
            "2016": "NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight",
            "2017": "NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight",
            "2018": "NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight",
        }
        # get scale factors
        values = {}
        values["nominal"] = self.cset[sfs_keys[self.year]].evaluate(
            self.pog_year, muon_eta, muon_pt, "sf"
        )
        values["up"] = self.cset[sfs_keys[self.year]].evaluate(
            self.pog_year, muon_eta, muon_pt, "systup"
        )
        values["down"] = self.cset[sfs_keys[self.year]].evaluate(
            self.pog_year, muon_eta, muon_pt, "systdown"
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"{self.tag}_triggeriso",
            weight=values["nominal"],
            weightUp=values["up"],
            weightDown=values["down"],
        )

    def add_weight(self, sf_type: str, working_point: str = "tight") -> None:
        """
        add muon ID (TightID) or Iso (LooseRelIso with mediumID) scale factors

        Parameters:
        -----------
            sf_type:
                Type of scale factor {'id', 'iso'}
            working_point:
                Working point {'medium', 'tight'}
        """
        # muon absolute pseudorapidity range: [0, 2.4)
        muon_eta = np.clip(self.muon_eta.copy(), 0.0, 2.399)

        # muon pt range: [15, 120)
        muon_pt = np.clip(self.muon_pt.copy(), 15.0, 119.999)

        # 'id' and 'iso' scale factors keys
        sfs_keys = {
            "id": "NUM_TightID_DEN_TrackerMuons"
            if working_point == "tight"
            else "NUM_MediumPromptID_DEN_TrackerMuons",
            "iso": "NUM_LooseRelIso_DEN_TightIDandIPCut"
            if working_point == "tight"
            else "NUM_LooseRelIso_DEN_MediumID",
        }
        # get scale factors
        values = {}
        values["nominal"] = self.cset[sfs_keys[sf_type]].evaluate(
            self.pog_year, muon_eta, muon_pt, "sf"
        )
        values["up"] = self.cset[sfs_keys[sf_type]].evaluate(
            self.pog_year, muon_eta, muon_pt, "systup"
        )
        values["down"] = self.cset[sfs_keys[sf_type]].evaluate(
            self.pog_year, muon_eta, muon_pt, "systdown"
        )
        # add scale factors to weights container
        self.weights.add(
            name=f"{self.tag}_{sf_type}",
            weight=values["nominal"],
            weightUp=values["up"],
            weightDown=values["down"],
        )


# ------------------------------
# OBJECT-WISE CORRECTIONS
# ------------------------------


# ------------------------------
# met phi modulation corrections
# ------------------------------
def met_phi_corrections(
    met_pt: ak.Array,
    met_phi: ak.Array,
    npvs: ak.Array,
    is_mc: bool,
    year: str,
    year_mod: str = "",
) -> Tuple[ak.Array, ak.Array]:
    """
    Apply MET phi corrections

    Parameters:
    -----------
        met_pt:
            MET transverse momentum
        met_phi:
            MET azimuthal angle
        npvs:
            Total number of reconstructed primary vertices
        is_mc:
            True if dataset is MC
        year:
            Year of the dataset {'2016', '2017', '2018'}
        year_mod:
            Year modifier {'', 'APV'}

    Returns:
    --------
        corrected MET pt and phi arrays
    """
    cset = correctionlib.CorrectionSet.from_file(
        get_pog_json(json_name="met", year=year)
    )
    # make sure to not cross the maximum allowed value for uncorrected met
    met_pt = np.clip(met_pt, 0.0, 6499.0)
    met_phi = np.clip(met_phi, -3.5, 3.5)

    run_ranges = {
        "2016APV": [272007, 278771],
        "2016": [278769, 284045],
        "2017": [297020, 306463],
        "2018": [315252, 325274],
    }

    data_kind = "mc" if is_mc else "data"
    run = np.random.randint(run_ranges[year][0], run_ranges[year][1], size=len(met_pt))

    try:
        corrected_met_pt = cset[f"pt_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), npvs.to_numpy(), run
        )
        corrected_met_phi = cset[f"phi_metphicorr_pfmet_{data_kind}"].evaluate(
            met_pt.to_numpy(), met_phi.to_numpy(), npvs.to_numpy(), run
        )

        return corrected_met_pt, corrected_met_phi
    except:
        return met_pt, met_phi


# -------------------
# JEC/JER corrections
# -------------------
# Recomendations https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECDataMC#Recommended_for_MC
def jet_corrections(events: NanoEventsArray, year: str) -> Tuple[ak.Array, ak.Array]:
    """
    Apply JEC/JER corrections to jets and MET.

    We use the script data/scripts/build_jec.py to create the 'mc_jec_compiled.pkl.gz'
    file with jet and MET factories

    Parameters:
    -----------
        events:
            events collection
        year:
            Year of the dataset {'2016', '2017', '2018'}
    """
    # load jet and MET factories with JEC/JER corrections
    with importlib.resources.path(
        "wprime_plus_b.data", "mc_jec_compiled.pkl.gz"
    ) as path:
        with gzip.open(path) as fin:
            factories = cloudpickle.load(fin)

    def add_jec_variables(jets: ak.Array, event_rho: ak.Array):
        """add some variables to the jet collection"""
        jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
        jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
        jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
        return jets

    # get corrected jets
    corrected_jets = factories["jet_factory"][year].build(
        add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll),
        events.caches[0],
    )

    # get corrected MET
    corrected_met = factories["met_factory"].build(events.MET, corrected_jets, {})

    return corrected_jets, corrected_met