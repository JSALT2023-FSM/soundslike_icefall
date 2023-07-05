# SoundsLike
Icefall recipe for the SoundsLike project under JSALT 2023.

## Installation and setup

First clone and install [icefall](https://github.com/k2-fsa/icefall). The following
instructions may be useful: <https://icefall.readthedocs.io/en/latest/installation/index.html>.

Then, from the root of the icefall repository, add this repository as a submodule:

```bash
git submodule add --name soundslike https://github.com/JSALT2023-FSM/soundslike_icefall.git egs/soundslike
```

This should set you up for running the recipes.

**NOTE:** For Lhotse installation, use the following command, since the master branch does
not have the VoxPopuli recipe yet:

```bash
pip install git+https://github.com/desh2608/lhotse.git@recipe/voxpopuli
```

### CMU pronunciation dictionary

To run the recipe using IPA, you will need to copy the dictionary to the `ASR/download` directory:

```bash
mkdir -p ASR/download
cp /data/soundslike/cmuipa.txt ASR/download/cmuipa.txt
```
