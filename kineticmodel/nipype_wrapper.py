import os
import nibabel as nib
from nipype.interfaces.base import TraitedSpec, DynamicTraitedSpec, File, traits, \
                                   BaseInterface, BaseInterfaceInputSpec, isdefined
from nipype.utils.filemanip import split_filename
from nipype.interfaces.io import add_traits

from temporalimage import load as ti_load
import kineticmodel

class KineticModelInputSpec(BaseInterfaceInputSpec):
    model = traits.Enum('SRTM_Zhou2003','SRTM_Lammertsma1996',
                        desc='one of: SRTM_Zhou2003, SRTM_Lammertsma1996',
                        mandatory=True)
    timeSeriesImgFile = File(exists=True, desc='Dynamic PET image', mandatory=True)
    frameTimingCsvFile = File(exists=True, desc='csv file listing the duration of each time frame in the 4D image, in minutes', mandatory=True)
    refRegionMaskFile = File(exists=True, desc='Reference region mask', mandatory=True)
    startTime = traits.Float(0.0, desc='minute into the time series image at which to start computing the parametric images, inclusive', mandatory=False)
    endTime = traits.Float(desc='minute into the time series image at which to stop computing the parametric images, exclusive', mandatory=False)

    time_unit = traits.Enum('min','s', desc='one of: min, s', mandatory=True)
    startActivity = traits.Enum('flat','increasing','zero', desc='one of: flat, increasing, zero', mandatory=True)
    weights = traits.Enum('frameduration','none','frameduration_activity','frameduration_activity_decay','trues',
                          desc='frameduration, none, frameduration_activity, frameduration_activity_decay, trues',
                          mandatory=True)
    halflife = traits.Float(desc='Halflife of the radiotracer. Must be provided in time_unit units (required for decay corrected weights)',
                            mandatory=False)
    #Trues = traits.array(desc="", mandatory=False)
    fwhm = traits.Float(desc='Full width at half max (in mm) for Gaussian smoothing (required for SRTM_Zhou2003)',
                        mandatory=False)

class KineticModel(BaseInterface):
    """
    Simplified Reference Tissue Model (SRTM) implemented using
    Linear Regression with Spatial Constraint (Zhou et al., 2003)

    """

    input_spec = KineticModelInputSpec
    output_spec = DynamicTraitedSpec

    def _run_interface(self, runtime):
        model = self.inputs.model
        timeSeriesImgFile = self.inputs.timeSeriesImgFile
        refRegionMaskFile = self.inputs.refRegionMaskFile
        frameTimingCsvFile = self.inputs.frameTimingCsvFile
        endTime = self.inputs.endTime

        ti = ti_load(timeSeriesImgFile, frameTimingCsvFile)

        if isdefined(self.inputs.startTime):
            startTime = self.inputs.startTime
        else:
            startTime = ti.get_startTime()

        if isdefined(self.inputs.endTime):
            endTime = self.inputs.EndTime
        else:
            endTime = ti.get_endTime()

        if isdefined(self.inputs.halflife):
            halflife = self.inputs.halflife
        else:
            halflife = None

        if isdefined(self.inputs.fwhm):
            fwhm = self.inputs.fwhm
        else:
            fwhm = None

        _, base, _ = split_filename(timeSeriesImgFile)

        ti = ti.extractTime(startTime, endTime)
        self.modStartTime = ti.get_startTime()
        self.modEndTime = ti.get_endTime()

        class_ = getattr(kineticmodel, model)
        results_img = class_.volume_wrapper(ti=ti,
                                            refRegionMaskFile=refRegionMaskFile,
                                            time_unit=self.inputs.time_unit,
                                            startActivity=self.inputs.startActivity,
                                            weights=self.inputs.weights,
                                            halflife=halflife,
                                            #Trues=self.inputs.Trues,
                                            fwhm=fwhm)

        for result_name in class_.result_names:
            res_img = nib.Nifti1Image(results_img[result_name], ti.affine, ti.header)
            res_fname = base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_'+result_name+'.nii.gz'
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
            outputs[result_name] = os.path.abspath(base+'_'+'{:02.2f}'.format(self.modEndTime)+'min_'+result_name+'.nii.gz')

        return outputs
