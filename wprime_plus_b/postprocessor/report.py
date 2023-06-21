import numpy as np
import pandas as pd


def build_report(
    accumulated_outputs: dict, xsecs: dict, lumi: float = 41477.877399
) -> pd.DataFrame:
    """
    build report with expected number of events and statistical
    errors for bkgs, data, total bkg, and data/total bkg
    """
    mcs = ["DYJetsToLL", "WJetsToLNu", "VV", "tt", "SingleTop"]
    events = {sample: 0 for sample in mcs}
    events.update({"Data": 0})
    errors = events.copy()

    for sample in accumulated_outputs:
        if ("SingleElectron" in sample) or ("SingleMuon" in sample):
            events["Data"] += accumulated_outputs[sample]["events_after"]
            errors["Data"] += np.sqrt(events["Data"])
            continue
        # get number of events before selection
        nevents = accumulated_outputs[sample]["events_before"]

        # get number of events after selection
        n_mc = accumulated_outputs[sample]["events_after"]

        # get expected number of events
        weight = (xsecs[sample] * lumi) / nevents
        n_phys = weight * n_mc

        # get statistical error
        error = weight * np.sqrt(n_mc)

        if "DYJetsToLL" in sample:
            events["DYJetsToLL"] += n_phys
            errors["DYJetsToLL"] += error
        elif "WJetsToLNu" in sample:
            events["WJetsToLNu"] += n_phys
            errors["WJetsToLNu"] += error
        elif ("WW" in sample) or ("WZ" in sample) or ("ZZ" in sample):
            events["VV"] += n_phys
            errors["VV"] += error
        elif "TTT" in sample:
            events["tt"] += n_phys
            errors["tt"] += error
        else:
            events["SingleTop"] += n_phys
            errors["SingleTop"] += error
            
    # add number of expected events and errors to report
    report_df = pd.DataFrame(columns=["events", "percentage", "error"])
    for sample in events:
        report_df.loc[sample, "events"] = events[sample]
        report_df.loc[sample, "error"] = errors[sample]
        
    # add percentages to report
    mcs_output = report_df.loc[mcs].copy()
    report_df.loc[mcs, "percentage"] = (
        mcs_output["events"] / mcs_output["events"].sum()
    ) * 100

    # (https://depts.washington.edu/imreslab/2011%20Lectures/ErrorProp-CountingStat_LRM_04Oct2011.pdf)
    # add total background number of expected events and error to report
    report_df.loc["Total bkg", "events"] = np.sum(report_df.loc[mcs, "events"])
    report_df.loc["Total bkg", "error"] = np.sqrt(
        np.sum(report_df.loc[mcs, "error"] ** 2)
    )

    # add data to bacground ratio and error
    data = report_df.loc["Data", "events"]
    data_err = report_df.loc["Data", "error"]
    bkg = report_df.loc["Total bkg", "events"]
    bkg_err = report_df.loc["Total bkg", "error"]

    report_df.loc["Data/bkg", "events"] = data / bkg
    report_df.loc["Data/bkg", "error"] = np.sqrt(
        (1 / bkg) ** 2 * data_err**2 + (data / bkg**2) ** 2 * bkg_err**2
    )

    return report_df