# specify the encoder types for pSp and e4e - this is mainly used for the inference alignment
ENCODER_TYPES = {
    'pSp': ['BackboneEncoder', 'ResNetBackboneEncoder'],
    'e4e': ['ProgressiveBackboneEncoder', 'ResNetProgressiveBackboneEncoder']
}
