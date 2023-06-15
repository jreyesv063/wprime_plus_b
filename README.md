# W' + b

[![Codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="left">
  <img width="300" src="https://i.imgur.com/OWhX13O.jpg" />
</p>

Python package for analyzing W' + b in the electron and muon channels. The analysis uses a columnar framework to process input tree-based [NanoAOD](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD) files using the [coffea](https://coffeateam.github.io/coffea/) and [scikit-hep](https://scikit-hep.org) Python libraries.

- [Processors](#Processors)
    * [Trigger efficiency erocessor](#Trigger-efficiency-processor)
    * [First tt control region processor](#First-tt-control-region-processor)
    * [Second tt control region processor](#Second-tt-control-region-processor)
- [How to run](#How-to-run)
    * [Submitting jobs at Coffea-Casa](#Submitting-jobs-at-Coffea-Casa)
    * [Submitting condor jobs at lxplus](#Submitting-condor-jobs-at-lxplus)
- [Scale factors](#Scale-factors)
- [Setting up coffea environments](#Setting-up-coffea-environments)
- [Data fileset](#Data-fileset)
    * [Re-making the input dataset files with DAS](#Re-making-the-input-dataset-files-with-DAS)
    * [Luminosity](#luminosity)

## Processors

### [Trigger efficiency](processors/trigger_efficiency_processor.py) 

Processor use to compute trigger efficiencies. 

The processor applies the following pre-selection cuts





| $$\textbf{Object}$$    | $$\textbf{Variable}$$          | $$\textbf{Cut}$$                                                    | 
| ---------------------  | ------------------------------ | ------------------------------------------------------------------- |
| $$\textbf{Electrons}$$ |                                |                                                                     |
|                        | $p_T$                        | $\geq 30$ GeV                                           |
|                        | $\eta$                       | $\| \eta \| < 1.44$ and $1.57 < \| \eta \| < 2.5$       |
|                        | pfRelIso04_all                 | $\lt 0.25$                                                        |
|                        | mvaFall17V2Iso_WP80 (ele) mvaFall17V2Iso_WP90 (mu) | $\text{True}$|
| $$\textbf{Muons}$$     |                                |                                                                     |
|                        | $p_T$                        | $\geq 30$ GeV                                          |
|                        | $\eta$                       | $\lt 2.4$                                                 |
|                        | pfRelIso04_all       | $\lt 0.25$                                                        |
|                        | mediumId (ele) tightId (mu) | $\text{True}$                   |
| $$\textbf{Jets}$$      |                                |                                                                     |
|                        | $p_T$                        |  $\geq 20$ GeV                                       |
|                        | $\eta$                       | $\lt 2.4$                                                 |
|                        | JetId                        | $6$                                                               |
|                        | puId                          | $7$                                                               |
|                        | btagDeepFlavB                | $\gt$ Medium WP                                          |



The trigger efficiency is computed as:


$$\epsilon = \frac{\text{selection cuts and reference trigger and main trigger}}{\text{selection cuts and reference trigger}}$$


so we define two regions for each channel: ```numerator``` and ```denominator```. We use the following triggers:


$\text{Analysis triggers}$


| Channel        | 2016           |    | 2017           |   | 2018           |
|----------------|----------------|--- |----------------|---|----------------|
| Muon           | IsoMu24        |    | IsoMu27        |   | IsoMu24        |
| Electron       | Ele27\_WPTight\_Gsf |   | Ele35\_WPTight\_Gsf |   | Ele32\_WPTight\_Gsf |


The reference and main triggers, alongside the selection criteria applied to establish each region, are presented in the following tables:


#### Electron channel

| Trigger        | 2016           |   | 2017           |   | 2018           |
|----------------|----------------|---|----------------|---|----------------|
| Reference trigger   | IsoMu24        |   | IsoMu27        |   | IsoMu24        |
| Main trigger         | Ele27\_WPTight\_Gsf |   | Ele35\_WPTight\_Gsf |   | Ele32\_WPTight\_Gsf |


| Selection cuts      | 
| ---------------------------------|
| Luminosity calibration            |
| MET filters                       |
| $N(bjet) \geq 1$                   |
| $N(\mu) = 1$                      |
| $N(e) = 1$                       |

#### Muon channel

| Trigger        | 2016           |   | 2017           |   | 2018           |
|----------------|----------------|---|----------------|---|----------------|
| Reference trigger         | Ele27\_WPTight\_Gsf |   | Ele35\_WPTight\_Gsf |   | Ele32\_WPTight\_Gsf |
| Main trigger   | IsoMu24        |   | IsoMu27        |   | IsoMu24        |



| Selection cuts      | 
| ------------------------------------------------------------------- |
| Luminosity calibration                                    |
| MET filters                        |
| $\Delta R (\mu, bjet) \gt 0.4$                           |
| $N(bjet) \geq 1$                           |
| $N(\mu) = 1$                        |
| $N(e) = 1$                       |


### [First tt control region](processors/ttbar_cr1_processor.py) 

Processor use to estimate backgrounds in a $t\bar{t}$ control region. 

The processor applies the following pre-selection cuts for the electron (ele) and muon (mu) channels:

| $$\textbf{Object}$$    | $$\textbf{Variable}$$          | $$\textbf{Cut}$$                                                    | 
| ---------------------  | ------------------------------ | ------------------------------------------------------------------- |
| $$\textbf{Electrons}$$ |                                |                                       |
|                        | $p_T$                        | $\geq 55$ GeV (ele)  $\geq 30$ GeV (mu)                                        |
|                        | $\eta$                       | $\| \eta \| < 1.44$ and $1.57 < \| \eta \| < 2.5$       |
|                        | pfRelIso04_all                 | $\lt 0.25$                                                        |
|                        | mvaFall17V2Iso_WP80 (ele) mvaFall17V2Iso_WP90 (mu) | $\text{True}$|
| $$\textbf{Muons}$$     |                                |                                                                     |
|                        | $p_T$                        | $\geq 35$ GeV                                         |
|                        | $\eta$                       | $\lt 2.4$                                                 |
|                        | pfRelIso04_all               | $\lt 0.25$                                                        |
|                        | tightId                      | $\text{True}$                   |
| $$\textbf{Taus}$$      |                                |                                                                     |
|                        | $p_T$                        | $\geq 20$ GeV                                               |
|                        | $\eta$                       | $\lt 2.3$                                                 |
|                        | $dz$                         | $\lt 0.2$                                                        | 
|                        | idDeepTau2017v2p1VSjet       | $\gt 8$                                                           |
|                        | idDeepTau2017v2p1VSe         | $\gt 8$                                                           |
|                        | idDeepTau2017v2p1VSmu        | $\gt 1$                                                           |
| $$\textbf{Jets}$$      |                                |                                                                     |
|                        | $p_T$                        |  $\geq 20$ GeV                                            |
|                        | $\eta$                       | $\lt 2.4$                                                 |
|                        | JetId                        | $6$                                                               |
|                        | puId                          | $7$                                                               |
|                        | btagDeepFlavB                | $\gt$ Medium WP                                          |




and additional selection cuts for each channel:

#### Electron channel

| Selection cuts      | 
| ---------------------------------|
| Electron Trigger      |
| Luminosity calibration                  |
| MET filters           |
| $p_T^{miss}\gt 50$ GeV |
| $N(bjet) = 2$                  |
| $N(\tau) = 0$                  |
| $N(\mu) = 0$                   |
| $N(e) = 1$                    |
| $\Delta R (e, bjet_0) \gt 0.4$ |

expected to be run with the `SingleElectron` dataset.

#### Muon channel


| Selection cuts      | 
| ---------------------------------|
| Muon Trigger          |
| Luminosity calibration                  |
| MET filters           |
| $p_T^{miss}\gt 50$ GeV |
| $N(bjet) = 2$                  |
| $N(\tau) = 0$                  |
| $N(e) = 0$                     |
| $N(\mu) = 1$                   |
| $\Delta R (\mu, bjet_0) \gt 0.4$ |

expected to be run with the `SingleMuon` dataset.

### [Second tt control region](processors/ttbar_cr2_processor.py) 

Processor use to estimate backgrounds in a $t\bar{t}$ control region. 

The processor applies the following pre-selection cuts for the electron (ele) and muon (mu) channels:

| $$\textbf{Object}$$    | $$\textbf{Variable}$$          | $$\textbf{Cut}$$                                                    | 
| ---------------------  | ------------------------------ | ------------------------------------------------------------------- |
| $$\textbf{Electrons}$$ |                                |                                       |
|                        | $p_T$                        | $\geq 55$ GeV (mu)  $\geq 30$ GeV (ele)                                        |
|                        | $\eta$                       | $\| \eta \| < 1.44$ and $1.57 < \| \eta \| < 2.5$       |
|                        | pfRelIso04_all                 | $\lt 0.25$                                                        |
|                        | mvaFall17V2Iso_WP80 (ele) mvaFall17V2Iso_WP90 (mu) | $\text{True}$|
| $$\textbf{Muons}$$     |                                |                                                                     |
|                        | $p_T$                        | $\geq 35$ GeV                                         |
|                        | $\eta$                       | $\lt 2.4$                                                 |
|                        | pfRelIso04_all               | $\lt 0.25$                                                        |
|                        | tightId                      | $\text{True}$                   |
| $$\textbf{Taus}$$      |                                |                                                                     |
|                        | $p_T$                        | $\geq 20$ GeV                                               |
|                        | $\eta$                       | $\lt 2.3$                                                 |
|                        | $dz$                         | $\lt 0.2$                                                        | 
|                        | idDeepTau2017v2p1VSjet       | $\gt 8$                                                           |
|                        | idDeepTau2017v2p1VSe         | $\gt 8$                                                           |
|                        | idDeepTau2017v2p1VSmu        | $\gt 1$                                                           |
| $$\textbf{Jets}$$      |                                |                                                                     |
|                        | $p_T$                        |  $\geq 20$ GeV                                            |
|                        | $\eta$                       | $\lt 2.4$                                                 |
|                        | JetId                        | $6$                                                               |
|                        | puId                          | $7$                                                               |
|                        | btagDeepFlavB                | $\gt$ Medium WP                                          |



and additional selection cuts for each channel:

#### Electron channel

| Selection cuts      | 
| ---------------------------------|
| Muon Trigger      |
| Luminosity calibration                  |
| MET filters           |
| $p_T^{miss}\gt 50$ GeV |
| $N(bjet) = 1$                  |
| $N(\tau) = 0$                  |
| $N(\mu) = 1$                   |
| $N(e) = 1$                    |
| $\Delta R (e, bjet_0) \gt 0.4$ |
| $\Delta R (\mu, bjet_0) \gt 0.4$ |

expected to be run with the `SingleMuon` dataset.

#### Muon channel


| Selection cuts      | 
| ---------------------------------|
| Electron Trigger          |
| Luminosity calibration                  |
| MET filters           |
| $p_T^{miss}\gt 50$ GeV |
| $N(bjet) = 1$                  |
| $N(\tau) = 0$                  |
| $N(e) = 1$                     |
| $N(\mu) = 1$                   |
| $\Delta R (\mu, bjet_0) \gt 0.4$ |
| $\Delta R (e, bjet_0) \gt 0.4$ |

expected to be run with the `SingleElectron` dataset.

## How to run

The `submit.py` file executes a desired processor with user-selected options. To see a list of arguments needed to run this script please enter the following in the terminal:

```bash
python3 submit.py --help
```
The output should look something like this:

```
usage: submit.py [-h] [--facility FACILITY] [--redirector REDIRECTOR] [--processor PROCESSOR] [--executor EXECUTOR] [--workers WORKERS] [--year YEAR] [--yearmod YEARMOD] [--channel CHANNEL]
                 [--fileset FILESET] [--sample SAMPLE] [--nfiles NFILES] [--nsplit NSPLIT] [--tag TAG] [--eos EOS]

optional arguments:
  -h, --help            show this help message and exit
  --facility FACILITY   facility to run jobs {'coffea-casa', 'lxplus'} (default coffea-casa)
  --redirector REDIRECTOR
                        redirector to find CMS datasets {use 'xcache' at coffea-casa. use 'cmsxrootd.fnal.gov', 'xrootd-cms.infn.it' or 'cms-xrd-global.cern.ch' at lxplus} (default xcache)
  --processor PROCESSOR
                        processor to be used {trigger, ttbar_cr1, ttbar_cr2, candle, btag_eff} (default ttbar_cr1)
  --executor EXECUTOR   executor to be used {iterative, futures, dask} (default iterative)
  --workers WORKERS     number of workers to use with futures executor (default 4)
  --year YEAR           year of the data {2016, 2017, 2018} (default 2017)
  --yearmod YEARMOD     year modifier {'', 'APV'} (default '')
  --channel CHANNEL     lepton channel to be processed {'mu', 'ele'} (default mu)
  --fileset FILESET     name of a json file at `wprime_plus_b/fileset` (default `wprime_plus_b/fileset/fileset_{year}_UL_NANO.json`)
  --sample SAMPLE       sample key to be processed {'all', 'mc' or <sample_name>} (default all)
  --nfiles NFILES       number of .root files to be processed by sample. To run all files use -1 (default 1)
  --nsplit NSPLIT       number of subsets to divide the fileset into (default 1)
  --tag TAG             tag of the submitted jobs (default test)
  --eos EOS             wheter to copy or not output files to EOS (default False)
```
By running this script, a desired processor is executed at some facility, defined by the `--facility` flag. [Coffea-Casa](https://coffea-casa.readthedocs.io/en/latest/cc_user.html) is faster and more convenient, however still somewhat experimental so for large inputs and/or processors which may require heavier cpu/memory using HTCondor at lxplus is recommended. 

You can select the executor to run the processor using the `--executor` flag. Three executors are available: `iterative`, `futures`, and `dask`. The `iterative` executor uses a single worker, while the `futures` executor uses the number of workers specified by the `--workers` flag. The `dask` executor uses Dask functionalities to scale up the analysis (only available at coffea-casa).

With `--fileset` you can define the name of a .json fileset at `wprime_plus_b/fileset`. By default, `--fileset UL`, selects the `wprime_plus_b/fileset/fileset_{year}_UL_NANO.json` fileset. The year can be selected using the `--year` flag, and the `--yearmod` flag is used to specify whether the dataset uses APV or not. Use `--sample all` or `--sample mc` to run over all or only MC samples, respectively. You can also select a particular sample with `--sample <sample_name>`, where the available sample names are:
  * `DYJetsToLL_M-50_HT-70to100`
  * `DYJetsToLL_M-50_HT-100to200`
  * `DYJetsToLL_M-50_HT-200to400`
  * `DYJetsToLL_M-50_HT-400to600`
  * `DYJetsToLL_M-50_HT-600to800`
  * `DYJetsToLL_M-50_HT-800to1200`
  * `DYJetsToLL_M-50_HT-1200to2500`
  * `DYJetsToLL_M-50_HT-2500toInf`
  * `WJetsToLNu_HT-100To200`
  * `WJetsToLNu_HT-200To400`
  * `WJetsToLNu_HT-400To600`
  * `WJetsToLNu_HT-600To800`
  * `WJetsToLNu_HT-800To1200`
  * `WJetsToLNu_HT-1200To2500`
  * `WJetsToLNu_HT-2500ToInf`
  * `ST_s-channel_4f_leptonDecays`
  * `ST_t-channel_antitop_5f_InclusiveDecays`
  * `ST_t-channel_top_5f_InclusiveDecays`
  * `ST_tW_antitop_5f_inclusiveDecays`
  * `ST_tW_top_5f_inclusiveDecays`
  * `TTTo2L2Nu`
  * `TTToHadronic`
  * `TTToSemiLeptonic`
  * `WW`
  * `WZ`
  * `ZZ`
  * `SingleElectron`
  * `SingleMuon`

To lighten the workload of jobs, the fileset can be divided into sub-filesets by means of the `--nsplit` flag. You can also define the number of `.root` files to use by sample using the `--nfiles` option. Set `--nfiles -1` to use all `.root` files. The `--tag` flag is used to defined a label for the submitted jobs.

When you attempt to open a CMS file, your application must query a redirector (defined by the `--redirector` flag) to find the file. Which redirector you use depends on your region and facility. At coffea-casa, use `--redirector xcache`. At lxplus, if you are working in the US, it is best to use `cmsxrootd.fnal.gov`, while in Europe and Asia, it is best to use `xrootd-cms.infn.it`. There is also a "global redirector" at `cms-xrd-global.cern.ch` which will query all locations.

### Submitting jobs at Coffea-Casa

Let's assume we are using coffea-casa and we want to execute the `ttbar_cr1` processor for the electron channel using the `TTTo2L2Nu` sample from 2017. To test locally first, can do e.g.:

```bash
python3 submit.py --processor ttbar_cr1 --executor iterative --channel ele --sample TTTo2L2Nu --nfiles 1 --tag test
```

To scale up the analysis using Dask, first you need to define your Dask client inside the `submit/submit_coffea_casa.py` script (line 37), and then type:

```bash
python3 submit.py --processor ttbar_cr1 --executor dask --channel ele --sample TTTo2L2Nu --nfiles -1 --nsplit 5 --tag test
```
The results will be stored in the `/outfiles` folder

### Submitting condor jobs at lxplus

To submit jobs at lxplus using HTCondor, you need to have a valid grid proxy in the CMS VO. This requires that you already have a grid certificate installed. The needed grid proxy is obtained via the usual command
```bash
voms-proxy-init --voms cms
```
To execute a processor using all samples of a particular year type:
```bash
python3 submit.py --processor ttbar_cr1 --facility lxplus --redirector cmsxrootd.fnal.gov --channel ele --sample all --year 2017 --nfiles -1 --nsplit 5 --tag test --eos True
```
The script will create the condor and executable files (using the `submit.sub` and `submit.sh` templates) needed to submit jobs, as well as the folders containing the logs and outputs within the `/condor` folder (click [here](https://batchdocs.web.cern.ch/local/quick.html) for more info). After submitting the jobs, you can watch their status typing
```bash
watch condor_q
```
If you set `--eos` to `True`, the logs and outputs will be copied to your EOS area. 

#### Notes: 
* Currently, the processors are only functional for the year 2017. 


## Scale factors

We use the common json format for scale factors (SF), hence the requirement to install [correctionlib](https://github.com/cms-nanoAOD/correctionlib). The SF themselves can be found in the central [POG repository](https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration), synced once a day with CVMFS: `/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration`. A summary of their content can be found [here](https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/). The SF implemented are (See [corrections](processors/corrections.py)):

* Pileup
* btagging
* Electron ID
* Electron Reconstruction
* Electron Trigger*
* Muon ID
* Muon Iso
* Muon Trigger (Iso)
* MET 

*The use of these scale factors are not by default approved from EGM. We are using the scale factors derived by Siqi Yuan https://twiki.cern.ch/twiki/bin/viewauth/CMS/EgHLTScaleFactorMeasurements


## Setting up coffea environments

#### Install miniconda (if you do not have it already)
In your lxplus area:
```
# download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# run and follow instructions  
bash Miniconda3-latest-Linux-x86_64.sh

# Make sure to choose `yes` for the following one to let the installer initialize Miniconda3
# > Do you wish the installer to initialize Miniconda3
# > by running conda init? [yes|no]
```
Verify the installation is successful by running conda info and check if the paths are pointing to your Miniconda installation. 
If you cannot run conda command, check if you need to add the conda path to your PATH variable in your bashrc/zshrc file, e.g.,
```
export PATH="$HOME/nobackup/miniconda3/bin:$PATH"
```
To disable auto activation of the base environment:
```
conda config --set auto_activate_base false
```

#### Set up a conda environment and install the required packages
```
# create a new conda environment
conda create -n coffea-env python=3.8

# activate the environment
conda activate coffea-env

# install packages
pip install numpy pandas coffea correctionlib pyarrow

# install xrootd
conda install -c conda-forge xrootd
```

## Data fileset

The fileset json files that contain a dictionary of the files per sample are in the `data/fileset` directory.

#### Re-making the input dataset files with DAS

```
# connect to lxplus with a port forward to access the jupyter notebook server
ssh <your_username>@lxplus.cern.ch localhost:8800 localhost:8800

# create a working directory and clone the repo (if you have not done yet)
git clone https://github.com/deoache/wprime_plus_b

# enable the coffea environment
conda activate coffea-env

# then activate your proxy
voms-proxy-init --voms cms --valid 100:00

# activate cmsset
source /cvmfs/cms.cern.ch/cmsset_default.sh

# open the jupyter notebook on a browser
cd data/fileset/
jupyter notebook --no-browser --port 8800
```

there should be a link looking like `http://localhost:8800/?token=...`, displayed in the output at this point, paste that into your browser.
You should see a jupyter notebook with a directory listing.

Open `filesetDAS.ipynb` and run it. The json files containing the datasets to be run should be saved in the same `fileset/` directory.

We use the recomended Run-2 UltraLegacy Datasets. See https://twiki.cern.ch/twiki/bin/view/CMS/PdmVRun2LegacyAnalysis


#### Luminosity

See luminosity recomendations for Run2 at https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2. To obtain the integrated luminosity type (on lxplus):

```
export PATH=$HOME/.local/bin:/afs/cern.ch/cms/lumi/brilconda-1.1.7/bin:$PATH
pip uninstall brilws
pip install --install-option="--prefix=$HOME/.local" brilws
```

* SingleMuon: type

```
brilcalc lumi -b "STABLE BEAMS" --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /fb -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt --hltpath HLT_IsoMu27_v*
```

output:
```
#Summary: 
+-----------------+-------+------+--------+-------------------+------------------+
| hltpath         | nfill | nrun | ncms   | totdelivered(/fb) | totrecorded(/fb) |
+-----------------+-------+------+--------+-------------------+------------------+
| HLT_IsoMu27_v10 | 13    | 36   | 8349   | 2.007255669       | 1.870333304      |
| HLT_IsoMu27_v11 | 9     | 21   | 5908   | 1.383159994       | 1.254273727      |
| HLT_IsoMu27_v12 | 47    | 122  | 46079  | 8.954672794       | 8.298296788      |
| HLT_IsoMu27_v13 | 91    | 218  | 124447 | 27.543983745      | 26.259684708     |
| HLT_IsoMu27_v14 | 2     | 13   | 4469   | 0.901025085       | 0.862255849      |
| HLT_IsoMu27_v8  | 2     | 3    | 1775   | 0.246872270       | 0.238466292      |
| HLT_IsoMu27_v9  | 11    | 44   | 14260  | 2.803797063       | 2.694566730      |
+-----------------+-------+------+--------+-------------------+------------------+
#Sum delivered : 43.840766620
#Sum recorded : 41.477877399
```

* SingleElectron: type

```
brilcalc lumi -b "STABLE BEAMS" --normtag /cvmfs/cms-bril.cern.ch/cms-lumi-pog/Normtags/normtag_PHYSICS.json -u /fb -i /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/Legacy_2017/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt --hltpath HLT_Ele35_WPTight_Gsf_v*
```
output:

```
#Summary: 
+--------------------------+-------+------+--------+-------------------+------------------+
| hltpath                  | nfill | nrun | ncms   | totdelivered(/fb) | totrecorded(/fb) |
+--------------------------+-------+------+--------+-------------------+------------------+
| HLT_Ele35_WPTight_Gsf_v1 | 2     | 3    | 1775   | 0.246872270       | 0.238466292      |
| HLT_Ele35_WPTight_Gsf_v2 | 11    | 44   | 14260  | 2.803797063       | 2.694566730      |
| HLT_Ele35_WPTight_Gsf_v3 | 13    | 36   | 8349   | 2.007255669       | 1.870333304      |
| HLT_Ele35_WPTight_Gsf_v4 | 9     | 21   | 5908   | 1.383159994       | 1.254273727      |
| HLT_Ele35_WPTight_Gsf_v5 | 20    | 66   | 22775  | 5.399580877       | 4.879405647      |
| HLT_Ele35_WPTight_Gsf_v6 | 27    | 56   | 23304  | 3.555091917       | 3.418891141      |
| HLT_Ele35_WPTight_Gsf_v7 | 93    | 231  | 128916 | 28.445008830      | 27.121940558     |
+--------------------------+-------+------+--------+-------------------+------------------+
#Sum delivered : 43.840766620
#Sum recorded : 41.477877399
```