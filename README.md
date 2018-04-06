[![CircleCI](https://circleci.com/gh/bilgelm/kineticmodel.svg?style=svg)](https://circleci.com/gh/bilgelm/kineticmodel)
[![codecov](https://codecov.io/gh/bilgelm/kineticmodel/branch/master/graph/badge.svg)](https://codecov.io/gh/bilgelm/kineticmodel)

# kineticmodel
Kinetic models for dynamic PET analysis

## Installation
Clone this repository to your machine, then type in terminal:
`pip install -e PATH_TO/kineticmodel`

If you'd like to have wrappers for integration with `nipype`, use the `nipype`
extra:
`pip install -e PATH_TO/kineticmodel[nipype]`

## To-do:

Must-to-have:
- [x] open PET data set to use in examples
- [x] `save_results` implementation in `kineticmodel.py`
- [ ] speed up SRTM_Lammertsma1996
- [ ] add examples for region-wise processing
- [ ] add examples for voxel-wise processing
- [ ] Additional tests to increase code coverage to 80%
- [ ] Improve ReadMe and donwload/install instructions

Nice-to-have:

- [ ] implementation of srtm2
- [x] wrapper function for voxelwise image processing
- [x] wrapper function for vertexwise image processing
- [x] `nipype` integration
- [ ] ❔ command line tools
- [ ] BIDS integration
- [ ] :question: Incorporate unit handling with a package like `pint` :beer:, `units`, `numericalunits`, or `astropy.units`
