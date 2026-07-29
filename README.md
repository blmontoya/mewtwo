# mewtwo
Final Project for CSCI 1430: Computer Vision

Semantic segmentation of LiDAR point clouds is a core
component of autonomous driving systems, enabling ve-
hicles to have real-time 3D scene understanding. In this
project, we implement RangeNet++, a dimensionality re-
duction approach that projects LiDAR point clouds onto 2D
range images. This allows standard convolutional neural
networks to perform semantic segmentation at a lower com-
putational cost that models working in 3D space. We train
and evaluate our implementation on the SemanticKITTI ur-
ban driving benchmark by using an encoder-decoder CNN
with Gaussian-weighted kNN post-processing to reduce re-
projection error. Due to computational constraints, training
was limited to 5 sequences over 70 epochs, requiring ap-
proximately 5 hours on OSCAR. After training, our model
achieved a peak mIoU of 25% compared to the original pa-
per’s 52.2% over 150 epochs and the full dataset. Our results
demonstrate that the implementation works well, as the loss
curve was still declining and suggesting that additional train-
ing time and data would increase performance. We then ana-
lyze per-class performance and discuss range image-based
segmentation, including reprojection error, class imbalance,
and trade-offs with dimensionality reduction.
