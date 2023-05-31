import json
from pathlib import Path

def divide_list(lst: list, n: int):
    """
    Divide a list into n sublists
    """
    size = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        if i < remainder:
            end = start + size + 1
        else:
            end = start + size
        result.append(lst[start:end])
        start = end
    return result


def divide_fileset(nsplits: int, fileset: str):
    main_dir = Path.cwd()
    fileset_path = Path(f"{main_dir}/wprime_plus_b/fileset")
    
    # load fileset  
    with open(f"{fileset_path}/{fileset}.json", "r") as handle:
        data = json.load(handle)

    # make output filesets directory
    output_directory = Path(f"{fileset_path}/filesets")
    if output_directory.exists():
        for file in output_directory.glob("*"):
            if file.is_file():
                file.unlink()
    else:
        output_directory.mkdir(parents=True)

    # split fileset and save filesets
    filesets = {}
    for sample in data.keys():
        keys = ".".join(f"{sample}_{i}" for i in range(1, nsplits + 1)).split(".")
        if nsplits == 1:
            keys = [k.rstrip("_1") for k in keys]
        values = divide_list(data[sample], nsplits) 

        for key, value in zip(keys, values):
            sample_data = {}
            sample_data[key] = list(value)

            filesets[key] = f"{output_directory}/{key}.json"
            with open(f"{output_directory}/{key}.json", "w") as json_file:
                json.dump(sample_data, json_file, indent=4, sort_keys=True)

    return filesets

def get_filesets(fileset: str, sample: str, year: str, nsplit: int):
    # divide filesets in args.nsplit json files
    fileset = f"fileset_{year}_UL_NANO" if fileset == "UL" else fileset
    filesets = divide_fileset(nsplit, fileset)
    if sample == "mc":
        filesets = {
            s: filename
            for s, filename in filesets.items()
            if s not in ["SingleElectron", "SingleMuon"]
        }
    elif sample not in ["all", "mc"]:
        filesets = {key: filesets[key] for key in filesets if sample in key}
    
    return filesets