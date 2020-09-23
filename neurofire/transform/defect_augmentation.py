import numpy as np
import vigra

from skimage.draw import line
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.morphology import binary_dilation

from inferno.utils.io_utils import yaml2dict
from inferno.io.transform import Transform
from .artifact_source import ArtifactSource


# This is inspired by
# https://github.com/funkey/gunpowder/blob/master/gunpowder/nodes/defect_augment.py
# NOTE: Superhuman Accuracy on SNEMI (https://arxiv.org/pdf/1706.00120.pdf)
# also applies defect augmentations:
# - Misalignment (applied to a single slice or a stack of slcies) -> similar to deform
# - Missing section -> missing slice
# - Out-of-focus section: apply a gaussian blur to some slices
class DefectAugmentation(Transform):
    """
    Augment raw data with transformations similar to defects common in TEM data.

    Arguments:
        p_missing_slice: probability for a missing slice
        p_low_contrast: probability for a low contrast slice
        p_deformed_slice: probaboloty for a deformed slice
        p_artifact_source: probability for inserting an artifact from data source
        ignore_slice_list: list with slices to ignore (i.e. not to augment)
        contrast_scale: scale of low contrast transformation
        deformation_mode: deformation mode that should be used
        deformation_strength: deformation strength in pixel
        artifact_source: data source for additional artifacts
        mean_val: mean value for artifact normalization
        std_val: std value for artifact normalization
    """
    def __init__(self, p_missing_slice, p_low_contrast,
                 p_deformed_slice, p_artifact_source=0,
                 ignore_slice_list=None, contrast_scale=0.1,
                 deformation_mode='undirected', deformation_strength=10,
                 artifact_source=None, mean_val=None, std_val=None,
                 **super_kwargs):
        super().__init__(**super_kwargs)

        # set the cumulative defect probabilities
        self.p_missing_slice = p_missing_slice
        self.p_low_contrast = self.p_missing_slice + p_low_contrast
        self.p_deformed_slice = self.p_low_contrast + p_deformed_slice
        self.p_artifact_source = self.p_deformed_slice + p_artifact_source
        assert self.p_artifact_source <= 1.

        self.ignore_slice_list = ignore_slice_list

        # set the parameters for deformation augments
        if isinstance(deformation_mode, str):
            assert deformation_mode in ('all', 'undirected', 'compress')
            self.deformation_mode = deformation_mode
        elif isinstance(deformation_mode, (list, tuple)):
            assert len(deformation_mode) == 2
            assert 'undirected' in deformation_mode
            assert 'compress' in deformation_mode
            self.deformation_mode = 'all'
        self.deformation_strength = deformation_strength

        # set the params for the artifact source
        if p_artifact_source != 0:
            assert isinstance(artifact_source, ArtifactSource), type(artifact_source)
            self.artifact_source = artifact_source
        self.contrast_scale = 0.1

        if mean_val is not None:
            assert std_val is not None
        self.mean_val = mean_val
        self.std_val = std_val

    def apply_missing_slice(self, section):
        # if we don't have a mean / std, we insert a black slice
        # otherwise, we insert random uniform sample
        if self.mean_val is None:
            return np.zeros_like(section, section.dtype)
        else:
            return np.round(
                np.random.normal(self.mean_val, self.std_val, size=section.shape)
            ).astype(section.dtype)

    def apply_low_contrast(self, section):
        mean = section.mean()
        section -= mean
        section *= self.contrast_scale
        section += mean
        return section

    # apply a deformation to the current slice, according to the modes:
    # - 'undirected': apply a random elastic transformation
    # - 'compress': compress the image towards a randomly generated line,
    #               similar to the compression defects
    def apply_deformed_slice(self, section):
        if self.deformation_mode in ('undirected', 'compress'):
            mode = self.deformation_mode
        else:
            mode = 'undireccted' if np.random.rand() < .5 else 'compress'

        if mode == 'compress':
            section = self.compress_section(section)
        else:
            section = self.undirected_deformation(section)
        return section

    # this simulates a typical defect:
    # missing line of data with rest of data compressed towards the line
    def compress_section(self, section):
        shape = section.shape
        # randomly choose fixed x or fixed y with p = 1/2
        fixed_x = np.random.rand() < .5
        if fixed_x:
            x0, y0 = 0, np.random.randint(1, shape[1] - 2)
            x1, y1 = shape[0] - 1, np.random.randint(1, shape[1] - 2)
        else:
            x0, y0 = np.random.randint(1, shape[0] - 2), 0
            x1, y1 = np.random.randint(1, shape[0] - 2), shape[1] - 1

        # generate the mask of the line that should be blacked out
        line_mask = np.zeros_like(section, dtype='bool')
        rr, cc = line(x0, y0, x1, y1)
        line_mask[rr, cc] = 1

        # generate vectorfield pointing towards the line to compress the image
        # first we get the unit vector representing the line
        line_vector = np.array([x1 - x0, y1 - y0], dtype='float32')
        line_vector /= np.linalg.norm(line_vector)
        # next, we generate the normal to the line
        normal_vector = np.zeros_like(line_vector)
        normal_vector[0] = - line_vector[1]
        normal_vector[1] = line_vector[0]

        # make meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # generate the vector field
        flow_x, flow_y = np.zeros_like(section), np.zeros_like(section)

        # find the 2 components where coordinates are bigger / smaller than the line
        # to apply normal vector in the correct direction
        components = vigra.analysis.labelImageWithBackground(
            np.logical_not(line_mask).view('uint8')
        )
        assert len(np.unique(components)) == 3, "%i" % len(np.unique(components))
        neg_val = components[0, 0] if fixed_x else components[-1, -1]
        pos_val = components[-1, -1] if fixed_x else components[0, 0]

        flow_x[components == pos_val] = self.deformation_strength * normal_vector[1]
        flow_y[components == pos_val] = self.deformation_strength * normal_vector[0]
        flow_x[components == neg_val] = - self.deformation_strength * normal_vector[1]
        flow_y[components == neg_val] = - self.deformation_strength * normal_vector[0]

        # add small random noise
        flow_x += np.random.uniform(-1, 1, shape) * (self.deformation_strength / 8.)
        flow_y += np.random.uniform(-1, 1, shape) * (self.deformation_strength / 8.)

        # apply the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)
        # print(flow_x.shape)
        # print(flow_y.shape)
        cval = 0.0 if self.mean_val is None else self.mean_val
        section = map_coordinates(section, (flow_y, flow_x),
                                  mode='constant', order=3, cval=cval).reshape(shape)

        # dilate the line mask and zero out the section below it
        line_mask = binary_dilation(line_mask, iterations=10)
        section[line_mask] = 0.
        return section

    def undirected_deformation(self, section):
        shape = section.shape

        # make meshgrid
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

        # generate random vector field and smooth it
        flow_x = np.random.uniform(-1, 1, shape) * self.deformation_strength
        flow_y = np.random.uniform(-1, 1, shape) * self.deformation_strength
        flow_x = vigra.gaussianSmoothing(flow_x, sigma=3.)  # sigma is hard-coded for now
        flow_y = vigra.gaussianSmoothing(flow_y, sigma=3.)  # sigma is hard-coded for now

        # apply the flow fields
        flow_x, flow_y = (x + flow_x).reshape(-1, 1), (y + flow_y).reshape(-1, 1)
        section = map_coordinates(section, (flow_y, flow_x), mode='constant').reshape(shape)
        return section

    def apply_artifact_source(self, section):

        # draw a random artifact location
        artifact_index = np.random.randint(len(self.artifact_source))
        artifact, alpha_mask = self.artifact_source[artifact_index]
        artifact = artifact.numpy().squeeze()
        alpha_mask = alpha_mask.numpy().squeeze()
        assert artifact.shape == section.shape,\
            "%s, %s" % (str(artifact.shape), str(section.shape))
        assert alpha_mask.shape == section.shape
        assert alpha_mask.min() >= 0., "%f" % alpha_mask.min()
        assert alpha_mask.max() <= 1., "%f" % alpha_mask.max()

        # blend the section raw data and the artifact according to the alpha mask
        section = section * (1. - alpha_mask) + artifact * alpha_mask
        return section

    def volume_function(self, tensor, z_offset=None):

        # we check for ignore slices if a z-offset is given and if we have
        # a ignore slice list
        have_ignore_slices = False
        if z_offset is not None and self.ignore_slice_list is not None:
            have_ignore_slices = True

        # we iterate over the slices and apply each defect trafo with the given probability
        for z in range(tensor.shape[0]):

            # check if this slice should be ignored
            if have_ignore_slices:
                if z + z_offset in self.ignore_slice_list:
                    continue
            r = np.random.random()

            if r < self.p_missing_slice:
                tensor[z] = self.apply_missing_slice(tensor[z])

            elif r < self.p_low_contrast:
                tensor[z] = self.apply_low_contrast(tensor[z])

            elif r < self.p_deformed_slice:
                tensor[z] = self.apply_deformed_slice(tensor[z])

            elif r < self.p_artifact_source:
                tensor[z] = self.apply_artifact_source(tensor[z])

        return tensor

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        p_missing_slice = config.get('p_missing_slice')
        p_low_contrast = config.get('p_low_contrast')
        p_deformed_slice = config.get('p_deformed_slice')
        p_artifact_source = config.get('p_artifact_source', 0)
        ignore_slice_list = config.get('ignore_slice_list', None)
        contrast_scale = config.get('contrast_scale', 0.1)
        deformation_mode = config.get('deformation_mode', 'undirected')
        deformation_strength = config.get('deformation_strength', 10)
        artifact_source_config = config.get('artifact_source', None)
        if artifact_source_config is not None:
            artifact_source = ArtifactSource.from_config(artifact_source_config)
        else:
            artifact_source = None
        return cls(p_missing_slice, p_low_contrast,
                   p_deformed_slice, p_artifact_source,
                   ignore_slice_list=ignore_slice_list,
                   contrast_scale=contrast_scale,
                   deformation_mode=deformation_mode,
                   deformation_strength=deformation_strength,
                   artifact_source=artifact_source)
