from luminoth import Detector, read_image, vis_objects
from pathlib import Path
from datetime import date


class Predictor:

    @staticmethod
    def predictor_object(
            checkpoint,
            image,
            save_image=False,
    ):
        prediction_objects = Detector(checkpoint=checkpoint).predict(read_image(image))
        image_output_path = str(Path.cwd() / 'images_output' / 'output-{date}.png').format(date=date.today())

        if save_image:
            vis_objects(
                read_image(image),
                prediction_objects,
            ).save(image_output_path)

            return prediction_objects

        return prediction_objects

