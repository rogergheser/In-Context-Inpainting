import pytorch_lightning as pl
from icm.models.in_context_matting import InContextMatting

class InContextImageSwapping(InContextMatting):
    def __init__(
        self,
        cfg_feature_extractor,
        cfg_in_context_decoder,
        cfg_loss_function,
        learning_rate,
        cfg_scheduler=None,
        **kwargs,
    ):
        super.__init__(
            cfg_feature_extractor,
            cfg_in_context_decoder,
            cfg_loss_function,
            learning_rate,
            cfg_scheduler,
            **kwargs,
        )

    def forward(self, reference_images, guidance_on_reference_image, source_images):
        feature_of_reference_image = self.feature_extractor.get_reference_feature(
            reference_images)

        feature_of_source_image = self.feature_extractor.get_source_feature(
            source_images)
        
        reference = {'feature': feature_of_reference_image,
                     'guidance': guidance_on_reference_image}

        source = {'feature': feature_of_source_image, 'image': source_images}

        output, cross_map, self_map = self.in_context_decoder(source, reference)
        ## features + matting done
        