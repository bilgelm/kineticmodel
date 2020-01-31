import os
import nibabel as nib
from nipype.interfaces.base import TraitedSpec, DynamicTraitedSpec, File, traits, \
                                   BaseInterface, BaseInterfaceInputSpec, isdefined
from nipype.utils.filemanip import split_filename
from nipype.interfaces.io import add_traits

from temporalimage import load as ti_load
from temporalimage import Quantity
import kineticmodel

class KineticModelInputSpec(BaseInterfaceInputSpec):
    model = traits.Enum(*kineticmodel.KineticModel.model_values, mandatory=True,
                        desc='one of: ' + \
                             ', '.join(kineticmodel.KineticModel.model_values))
    timeSeriesImgFile = File(exists=True, mandatory=True,
                             desc='path to dynamic PET image')
    frameTimingFile = File(exists=True, mandatory=True,
                              desc=('csv/sif/json file listing the duration of '
                                    'each time frame in the 4D image'))
    refRegionMaskFile = File(exists=True, mandatory=True,
                             desc='Reference region mask')
    startTime = traits.Float(0.0, mandatory=False,
                             desc=('minute into the time series image at which '
                         'to start computing the parametric images, inclusive'))
    endTime = traits.Float(mandatory=False,
                           desc=('minute into the time series image at which '
                          'to stop computing the parametric images, exclusive'))

    startActivity = traits.Enum(*kineticmodel.KineticModel.startActivity_values,
                                mandatory=True,
                                desc='one of: ' + \
                      ', '.join(kineticmodel.KineticModel.startActivity_values))
    weights = traits.Enum(*kineticmodel.KineticModel.weights_values,
                          desc='one of: ' + \
                          ', '.join(kineticmodel.KineticModel.weights_values),
                          mandatory=True)
    halflife = traits.Float(mandatory=False,
                            desc=('halflife of the radiotracer, in minutes '
                                  '(required for decay corrected weights)'))
    fwhm = traits.Float(mandatory=False,
                        desc=('Full width at half max (in mm) for Gaussian '
                              'smoothing (required for SRTM_Zhou2003)'))

class KineticModel(BaseInterface):
    """
    Kinetic model applied to voxelwise data.
    """

    input_spec = KineticModelInputSpec
    output_spec = DynamicTraitedSpec

    def _run_interface(self, runtime):
        model = self.inputs.model
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        refRegionMaskFile = self.inputs.refRegionMaskFile
        frameTimingFile = self.inputs.frameTimingFile
        endTime = self.inputs.endTime

        ti = ti_load(timeSeriesImgFile, frameTimingFile)

        if isdefined(self.inputs.startTime):
            startTime = Quantity(self.inputs.startTime, 'minute')
        else:
            startTime = ti.get_startTime()

        if isdefined(self.inputs.endTime):
            endTime = Quantity(self.inputs.endTime, 'minute')
        else:
            endTime = ti.get_endTime()

        if isdefined(self.inputs.halflife):
            halflife = Quantity(self.inputs.halflife, 'minute')
        else:
            halflife = None

        if isdefined(self.inputs.fwhm):
            fwhm = self.inputs.fwhm
        else:
            fwhm = None

        _, base, _ = split_filename(timeSeriesImgFile)

        ti = ti.extractTime(startTime, endTime)
        self.modStartTime = ti.get_startTime().to('min').magnitude
        self.modEndTime = ti.get_endTime().to('min').magnitude

        class_ = getattr(kineticmodel, model)
        results_img = class_.volume_wrapper(ti=ti,
                                            refRegionMaskFile=refRegionMaskFile,
                                            startActivity=self.inputs.startActivity,
                                            weights=self.inputs.weights,
                                            halflife=halflife,
                                            fwhm=fwhm)

        for result_name in class_.result_names:
            res_img = nib.Nifti1Image(results_img[result_name],
                                      ti.affine, ti.header)
            res_fname = base + '_' + '{:02.2f}'.format(self.modEndTime) + \
                        'min_'+result_name+'.nii.gz'
            nib.save(res_img,res_fname)

        return runtime

    def _add_output_traits(self, base):
        class_ = getattr(kineticmodel, self.inputs.model)
        return add_traits(base, class_.result_names)

    def _outputs(self):
        return self._add_output_traits(super(KineticModel, self)._outputs())

    def _list_outputs(self):
        outputs = self._outputs().get()
        model = self.inputs.model
        fname = self.inputs.timeSeriesImgFile
        _, base, _ = split_filename(fname)

        class_ = getattr(kineticmodel, model)

        for result_name in class_.result_names:
            outputs[result_name] = os.path.abspath(base + '_' + \
                '{:02.2f}'.format(self.modEndTime)+'min_'+result_name+'.nii.gz')

        return outputs



class KineticModelROIInputSpec(BaseInterfaceInputSpec):
    model = traits.Enum(*kineticmodel.KineticModel.model_values, mandatory=True,
                        desc='one of: ' + \
                             ', '.join(kineticmodel.KineticModel.model_values))
    roiTACcsvFile = File(exists=True, mandatory=True,
                         desc='csv file containing TACs per ROI')
    frameTimingFile = File(exists=True, mandatory=True,
                              desc=('csv/sif/json file listing the duration of '
                                    'each time frame in the 4D image'))
    refRegion = traits.String(desc=('Name of reference region, ',
                                    'must be included in roiTACcsvfile'),
                              mandatory=True)
    startTime = traits.Float(0.0, mandatory=False,
                             desc=('minute into the time series image at which '
                                   'to start computing the parametric images, '
                                   'inclusive'))
    endTime = traits.Float(desc=('minute into the time series image at which '
                                 'to stop computing the parametric images, '
                                 'exclusive'),
                           mandatory=False)

    startActivity = traits.Enum(*kineticmodel.KineticModel.startActivity_values,
                                mandatory=True,
                                desc='one of: ' + \
                      ', '.join(kineticmodel.KineticModel.startActivity_values))
    weights = traits.Enum(*kineticmodel.KineticModel.weights_values,
                          desc='one of: ' + \
                          ', '.join(kineticmodel.KineticModel.weights_values),
                          mandatory=True)
    halflife = traits.Float(mandatory=False,
                            desc=('halflife of the radiotracer, in minutes '
                                  '(required for decay corrected weights)'))

class KineticModelROIOutputSpec(TraitedSpec):
    csvFile = File(exists=True, desc='csv file')

class KineticModelROI(BaseInterface):
    """
    Kinetic model applied to regional data.
    """

    input_spec = KineticModelROIInputSpec
    output_spec = KineticModelROIOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd
        from temporalimage.t4d import _csvread_frameTiming, \
                                      _sifread_frameTiming, \
                                      _jsonread_frameTiming

        model = self.inputs.model
        roiTACcsvFile = self.inputs.roiTACcsvFile
        refRegion = self.inputs.refRegion
        frameTimingFile = self.inputs.frameTimingFile
        endTime = self.inputs.endTime

        roiTACs = pd.read_csv(roiTACcsvFile)

        _, timingfileext = os.path.splitext(frameTimingFile)
        if timingfileext=='.csv':
            frameStart, frameEnd = _csvread_frameTiming(frameTimingFile)
        elif timingfileext=='.sif':
            frameStart, frameEnd, _ = _sifread_frameTiming(frameTimingFile)
        elif timingfileext=='.json':
            frameStart, frameEnd, _ = _jsonread_frameTiming(frameTimingFile)

        # Compute the time mid-way for each time frame
        t = (frameStart + frameEnd ) / 2

        # Compute the duration of each time frame
        dt = frameEnd - frameStart

        if isdefined(self.inputs.startTime):
            startTime = Quantity(self.inputs.startTime, 'minute')
        else:
            startTime = frameStart[0]

        if isdefined(self.inputs.endTime):
            endTime = Quantity(self.inputs.endTime, 'minute')
        else:
            endTime = frameEnd[-1]

        if isdefined(self.inputs.halflife):
            halflife = Quantity(self.inputs.halflife, 'minute')
        else:
            halflife = None

        _, base, _ = split_filename(roiTACcsvFile)

        # find the first time frame with frameStart at or shortest after the specified start time
        startIndex = next((i for i,t in enumerate(frameStart) if t>=startTime), len(frameStart)-1)
        # find the first time frame with frameEnd shortest after the specified end time
        endIndex = next((i for i,t in enumerate(frameEnd) if t>endTime), len(frameStart))

        TAC_rownames = roiTACs['ROI'].values

        isref = TAC_rownames==refRegion
        if isref.sum()!=1:
            raise ValueError("Exactly one row should correspond to the reference TAC")

        # separate reference region TAC from other TACs
        # we add 1 to startIndex and endIndex because the first column in
        # roiTACs is ROI names
        refTAC = roiTACs.iloc[isref,startIndex+1:endIndex+1].values.flatten()
        TAC = roiTACs.iloc[~isref,startIndex+1:endIndex+1].values
        TAC_rownames = TAC_rownames[~isref]

        # subset time vectors
        t = t[startIndex:endIndex]
        dt = dt[startIndex:endIndex]

        class_ = getattr(kineticmodel, model)
        km = class_(t, dt, TAC, refTAC,
                    startActivity=self.inputs.startActivity,
                    weights=self.inputs.weights,
                    halflife=halflife)
        km.fit()

        results = pd.DataFrame({'ROI': TAC_rownames})
        for result_name in class_.result_names:
            results[result_name] = km.results[result_name]

        results.to_csv(base+'_'+model+'_results.csv', index=False)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        _, base, _ = split_filename(self.inputs.roiTACcsvFile)
        outputs['csvFile'] = os.path.abspath(base+'_'+self.inputs.model+'_results.csv')

        return outputs
