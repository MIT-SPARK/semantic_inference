# Message containing an image masked by instance ids and associated feature vectors

Header header                                       # Header of image that the features were sourced from
int32[] mask_ids                                    # Mask IDs if non-sequential (should be empty otherwise)
sensor_msgs/Image image                             # Masked image by segment/feature ID
semantic_inference_msgs/FeatureVector[] features    # Feature vectors for each segment. Size must be equal to the number of masks
