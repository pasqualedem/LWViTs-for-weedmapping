from data.sequoia import SequoiaDataset
from keras_segmentation.models.segnet import segnet


def train():
    model = segnet(n_classes=3, input_height=480, input_width=360)
    data = SequoiaDataset("dataset/Sequoia", batch_size=4)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.summary()
    model.fit(data)


if __name__ == '__main__':
    train()