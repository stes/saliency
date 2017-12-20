# Visual Saliency: From Classical to Modern Approaches

*This repository was developed as part of a research lab during the [M.Sc. Neuroenginering program](msne.ei.tum.de). It does not contain original research besides slight modifications of the original code*

Predicting the ways in which image locations draw the attention of humans gives important insights into the visual system and the way in which humans access image contents.
The notion of saliency is a popular research item in both neuroscience and lately, also in deep learning research.
As attention based gating is incorporated in signal processes, predicting saliency to identify important parts of the image is also interesting from a techical perspective.
In this report, the implementation of a variant of the Itty Koch model is discussed as an example for a historical approach to saliency.
The approach will be compared to a simple data-driven adaptation approach as well as the state-of-the art model for visual saliency, DeepGazeII according to the MIT300 benchmark.
The models will be evaluated on a custom dataset of four photographs, three artificial visual stimuli as well as three video files.  

## Python package

The saliency models are implemented in Python.
Requirements for running the models are:

```
- numpy
- scipy
- pandas
- tensorflow (only for Deep Gaze II and ICF)
```

## References

The MIT300 and CAT2000 datasets can be obtained from the homepage of the [MIT Saliency Benchmark](http://saliency.mit.edu).
The implementation of Deep Gaze II and ICF from the [Deep Gaze](https://deepgaze.bethgelab.org) implementation.
Further information can be found in the papers

- Matthias Kümmerer, Thomas S.A. Wallis, Matthias Bethge: Understanding Low- and High-Level Contributions to Fixation Prediction ICCV 2017
- Matthias Kümmerer, Thomas S.A. Wallis, Matthias Bethge: Saliency Benchmarking: Separating Models, Maps and Metrics arXiv:1704.08615
- Matthias Kümmerer, Lucas Theis, Matthias Bethge: Deep Gaze I: Boosting Saliency Prediction with Feature Maps Trained on ImageNet (ICLR 2015 workshop paper)

## Contact

For any further inquires, please find my contact information on [my personal homepage](http://stes.io).
