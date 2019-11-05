from imagelib import RankSRGAN


class TrainableRankSRGAN(RankSRGAN):
    def __init__(self):
        super(TrainableRankSRGAN, self).__init__()

    def load_weights(self, model_filename):
        self.model.load_weights(model_filename)

    def save_weights(self, model_filename):
        self.model.save_weights(model_filename)

    def train(self):
        pass

