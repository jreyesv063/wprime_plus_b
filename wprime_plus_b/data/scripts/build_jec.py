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
    ext.add_weight_sets([f"* * {data_path}/{file}" for file in files])
    ext.finalize()
    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


jet_factory = {
    "2017mc": jet_factory_factory(
        files=[
            # https://github.com/cms-jet/JRDatabase/raw/master/textFiles/Fall17_V3b_MC/Fall17_V3b_MC_PtResolution_AK4PFchs.txt
            "Fall17_V3b_MC_PtResolution_AK4PFchs.jr.txt.gz",
            # https://github.com/cms-jet/JRDatabase/raw/master/textFiles/Fall17_V3b_MC/Fall17_V3b_MC_SF_AK4PFchs.txt
            "Fall17_V3b_MC_SF_AK4PFchs.jersf.txt.gz",
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