from data.data_sets import images_data_set
from predictor import Predictor
from pathlib import Path
import numpy
import time


class TestPerceptionResults:
    CHECKPOINT = 'accurate'
    IMAGE_DATA_SET = images_data_set['people']
    SAVE_IMAGE_OUTPUT = True

    @classmethod
    def setup_class(cls):
        people_predictor = Predictor()

        cls.actual_people_object = people_predictor.predictor_object(
            checkpoint=cls.CHECKPOINT,
            image=cls.IMAGE_DATA_SET,
            save_image=cls.SAVE_IMAGE_OUTPUT
        )

        if cls.CHECKPOINT is 'accurate':
            cls.expected_people_object = numpy.load(
                str(Path.cwd() / 'data' / 'accurate_people_object.npy'),
                allow_pickle=True
            ).tolist()
        else:
            cls.expected_people_object = numpy.load(
                str(Path.cwd() / 'data' / 'fast_people_object.npy'),
                allow_pickle=True
            ).tolist()

    def test_people_object_equality(self, benchmark):
        benchmark(time.sleep, 0.000001)

        assert numpy.array_equal(
            self.expected_people_object,
            self.actual_people_object,
        )

    def test_num_of_people(self, benchmark):
        benchmark(time.sleep, 0.000001)
        accurate_num_of_people = 4
        fast_num_of_people = 7

        people_sum = [
            1 for person
            in self.expected_people_object
            if 'person' in person['label']
        ]

        if self.CHECKPOINT is 'accurate':
            assert sum(people_sum) == accurate_num_of_people
        assert sum(people_sum) == fast_num_of_people
