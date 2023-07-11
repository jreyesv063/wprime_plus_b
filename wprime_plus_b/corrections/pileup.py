import correctionlib
import awkward as ak
from typing import Type
from coffea.analysis_tools import Weights
from wprime_plus_b.corrections.utils import get_pog_json


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