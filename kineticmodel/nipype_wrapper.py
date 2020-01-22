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
    Simplified Reference Tissue Model (SRTM) implemented using
    Linear Regression with Spatial Constraint (Zhou et al., 2003)

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
        self.modStartTime = ti.get_startTime()
        self.modEndTime = ti.get_endTime()

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
