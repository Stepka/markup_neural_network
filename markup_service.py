# -*- coding: utf-8 -*-
import markup_model


def init(config, start_stage=4):

    markup_model.init(config, start_stage)


def get_markup(texts=None):
    if markup_model.dnn_classifier_model is None:
        markup_model.init("rusvectores.config", 4)

    classes, probabilities, texts = markup_model.predict(markup_model.dnn_classifier_model, texts)

    result = {}
    for i in range(len(classes)):
        if not classes[i] in result:
            result[int(classes[i])] = []
        result[int(classes[i])].append((texts[i], round((probabilities[i]), 3)))

    return result
