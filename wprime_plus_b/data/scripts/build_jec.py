import importlib.resources
import contextlib
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory

jec_name_map = {
    'JetPt': 'pt',
    'JetMass': 'mass',
    'JetEta': 'eta',
    'JetA': 'area',
    'ptGenJet': 'pt_gen',
    'ptRaw': 'pt_raw',
    'massRaw': 'mass_raw',
    'Rho': 'event_rho',
    'METpt': 'pt',
    'METphi': 'phi',
    'JetPhi': 'phi',
    'UnClusteredEnergyDeltaX': 'MetUnclustEnUpDeltaX',
    'UnClusteredEnergyDeltaY': 'MetUnclustEnUpDeltaY',
}


def jet_factory_factory(files):
    data_path = "/home/cms-jovyan/wprime_plus_b/wprime_plus_b/data/"
    ext = extractor()
    ext.add_weight_sets([
        f"* * {data_path}/{file}" for file in files
    ])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/JECDataMC#Recommended_for_MC
jet_factory = {
    "2017mc": jet_factory_factory(
        files=[
            # JEC: https://github.com/cms-jet/JECDatabase/tree/master/textFiles
            "Summer19UL17_V5_MC_L1FastJet_AK4PFchs.jec.txt",
            "Summer19UL17_V5_MC_L2L3Residual_AK4PFchs.jec.txt",
            "Summer19UL17_V5_MC_L2Relative_AK4PFchs.jec.txt",
            "Summer19UL17_V5_MC_L3Absolute_AK4PFchs.jec.txt",
            "RegroupedV2_Summer19UL17_V5_MC_UncertaintySources_AK4PFchs.junc.txt",
            # JER: https://github.com/cms-jet/JRDatabase/tree/master/tarballs
            "Summer19UL17_JRV2_MC_PtResolution_AK4PFchs.jr.txt",
            "Summer19UL17_JRV2_MC_SF_AK4PFchs.jersf.txt",
        ]
    )
}
met_factory = CorrectedMETFactory(jec_name_map)


if __name__ == "__main__":
    import sys
    import gzip
    # jme stuff not pickleable in coffea
    import cloudpickle

    with gzip.open("/home/cms-jovyan/wprime_plus_b/wprime_plus_b/data/jec_compiled.pkl.gz", "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
                "met_factory": met_factory,
            },
            fout
        )