import correctionlib
import numpy as np
import awkward as ak
from typing import Tuple
from wprime_plus_b.corrections.utils import get_pog_json


def met_phi_corrections(
    met_pt: ak.Array,
    met_phi: ak.Array,
    npvs: ak.Array,
    is_mc: bool,
    year: str,
    year_mod: str = "",
) -> Tuple[ak.Array, ak.Array]:
    """
    Apply MET phi modulation corrections

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
        corrected MET pt and phi
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