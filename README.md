# real-salient

This is a header-only library that implements a modified version of
the [GrabCut algorithm](https://en.wikipedia.org/wiki/GrabCut).
The problem it solves is the two-class image segmentation (foreground/background);
in other words, it detects a salient object in an RGB-D image.
It follows the logic described in the DenseCut paper by Cheng et al[[1]](#1),
adapting it to GPU.

In short, the algorithm performs the following for every frame:

  1. Get the color and depth buffers from a depth camera.
  2. Assuming the salient object is in front, label the image pixels based on a simple treshold and the depth buffer
     (i.e. just like in the [librealsense-grabcuts](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/opencv/grabcuts) example).
  3. Fit two Gaussian Mixture Models (GMM) onto the color frame to create the color models of background and foreground.
  4. Use the trained models to label the image.
  5. Use a Conditional Random Field model (CRF) to refine the labels.

__Steps  2-5 are performed entirely on GPU, which allowed me to run the algorithm at steady 30 FPS.__

#### Gaussian Mixture Models

The `gmm.cuh` module is a generic CUDA implementation of the
[GMM](https://en.wikipedia.org/wiki/Mixture_model#Multivariate_Gaussian_mixture_model).
It can fit `M` GMMs, `K` components each, on a single image at once.
Thus, this module alone can be used for realtime `M`-class image segmentation.

The module uses the standard
[EM-algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm#Gaussian_mixture)
for estimation and
[Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition#LDL_decomposition_2)
for computing the covariance inverse and determinant.


#### Conditional Random Fields

The `crf.cuh` module is an adaptation of the GPU implementation of CRF by
[Jiahui Huang](https://github.com/heiwang1997/DenseCRF),
who used [Miguel Monteiro's](https://github.com/MiguelMonteiro/permutohedral_lattice)
implementation of fast gaussian filtering.
The theory for this implementation can be found in [[2]](#2) and [[3]](#3).

## Examples

The examples make use of the core `real-salient` as well as of couple VR-related tricks.
They are hardcoded to use the Intel RealSense D415 camera and its SDK to
capture a color+depth video stream (`examples/vr-salient/include/cameraD415.hpp`).

__VR bounds:__
The examples use [OpenVR](https://github.com/ValveSoftware/openvr) to improve the initial guess
of the salient object position (step 2 in the algorithm above).
I attach an extra tracker to the depth camera to locate its position in VR.
This allows me to find the position of the headset and hand controllers on the image via a simple coordinate transform.
This is implemented in `examples/vr-salient/include/vrbounds.hpp`.

__VR depth stencil:__
In addition to the tracker positions, I employ a `Vulkan+OpenVR` combination to render the VR shaperone bounds
into a temporary buffer.
This allows me to cut-off all objects outside the user-defined play area from the scene.
This is implemented in `examples/vr-salient/include/vulkanheadless.hpp`.
 

### vr-salient

`vr-salient` is a standalone program.
In addition to the tweaks above, it uses OpenCV highgui library - only to display the window.
The VR tricks are optional in this example.

### saber-salient

`saber-salient` is a dynamic library to be used in my [BeatSaber plugin](https://github.com/achirkin/CameraPlus).
It functions the same as `vr-salient`, but does not require `OpenCV` and requires VR tracking.

#### Demo videos:

[<img src="examples/saber-salient/img/screen-1.jpg?raw=true" width="260" alt="360° Reason for Living by Morgan Page" title="360° Reason for Living by Morgan Page"/>](https://youtu.be/1GdDrsxVWYE)

[<img src="examples/saber-salient/img/screen-2.jpg?raw=true" width="260" alt="360° First of the Year (Equinox) by Skrillex" title="360° First of the Year (Equinox) by Skrillex"/>](https://youtu.be/0zMn-zVGNNc)




## References

<a id="1" href="https://doi.org/10.1111/cgf.12758">[1]</a>
[[pdf]](http://mftp.mmcheng.net/Papers/DenseCut.pdf)
Cheng, M.M., Prisacariu, V.A., Zheng, S., Torr, P.H.S. and Rother, C.
_DenseCut: Densely Connected CRFs for Realtime GrabCut._
Computer Graphics Forum, 34: 193-201. 2015.

<a id="2" href="https://arxiv.org/abs/1210.5644">[2]</a>
Krähenbühl, Philipp, and Vladlen Koltun.
_Efficient inference in fully connected crfs with gaussian edge potentials._
Advances in neural information processing systems. 2011.

<a id="3" href="https://graphics.stanford.edu/papers/permutohedral">[3]</a>
Adams, Andrew, Jongmin Baek, and Myers Abraham Davis.
_Fast high‐dimensional filtering using the permutohedral lattice._
Computer Graphics Forum. Vol. 29. No. 2. Oxford, UK: Blackwell Publishing Ltd, 2010.
