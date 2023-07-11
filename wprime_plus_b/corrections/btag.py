import json
import correctionlib
import numpy as np
import awkward as ak
import importlib.resources
from coffea import util
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json, clip_array


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
        pass_mask = self._jet_map[flavor]["btagDeepFlavB"] > self._btagwp
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