import json
from pathlib import Path

def divide_list(lst, n):
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


def divide_fileset(samples, nsplits, year=2017):
    main_dir = Path.cwd()
    fileset_path = Path(f"{main_dir}/wprime_plus_b/fileset")
    data_path = Path(f"{main_dir}/wprime_plus_b/data")
    
    # make output filesets directory
    output_directory = Path(f"{fileset_path}/filesets")
    if output_directory.exists():
        for file in output_directory.glob("*"):
            if file.is_file():
                file.unlink()
    else:
        output_directory.mkdir(parents=True)
        
    # load simplified names for datasets
    with open(f"{data_path}/simplified_samples.json", "r") as handle:
        simplified_samples = json.load(handle)[year]
            
    # load fileset  
    with open(f"{fileset_path}/fileset_{year}_UL_NANO.json", "r") as handle:
        data = json.load(handle)
    
    # split fileset and save filesets
    filesets = {}
    for sample in samples:
        keys = ".".join(f"{simplified_samples[sample]}_{i}" for i in range(1, nsplits + 1)).split(".")
        values = divide_list(data[sample], nsplits) 

        for key, value in zip(keys, values):
            sample_data = {}
            sample_data[key] = list(value)
            
            filesets[key] = f"{output_directory}/{key}.json"
            with open(f"{output_directory}/{key}.json", "w") as json_file:
                json.dump(sample_data, json_file, indent=4, sort_keys=True)

    return filesets