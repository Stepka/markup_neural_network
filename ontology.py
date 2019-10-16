# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import gensim
import numpy as np
from pandas_ods_reader import read_ods
from ufal.udpipe import Model, Pipeline
import os
import sys
import argparse
import wget
import random
from time import time
import re
import shutil

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import joblib
import json
from collections import defaultdict

import multiprocessing
import logging  # Setting up the loggings to monitor gensim

# stage from pipeline we starting with. 4 - start from predictions
START_STAGE = 4
config = {}
data_root_path = "data/"
saved_models_path = "saved_models/"

word2vec = None
kmeans = None
dnn_classifier_model = None


def log(*texts):
    print(*texts)
    logging.info(" ".join(str(x) for x in texts))


def num_replace(word):
    new_token = 'x' * len(word)
    return new_token


def clean_token(token, misc):
    """
    :param token:  токен (строка)
    :param misc:  содержимое поля "MISC" в CONLLU (строка)
    :return: очищенный токен (строка)
    """
    out_token = token.strip().replace(' ', '')
    if token == 'Файл' and 'SpaceAfter=No' in misc:
        return None
    return out_token


def clean_lemma(lemma, pos):
    """
    :param lemma: лемма (строка)
    :param pos: часть речи (строка)
    :return: очищенная лемма (строка)
    """
    out_lemma = lemma.strip().replace(' ', '').replace('_', '').lower()
    if '|' in out_lemma or out_lemma.endswith('.jpg') or out_lemma.endswith('.png'):
        return None
    if pos != 'PUNCT':
        if out_lemma.startswith('«') or out_lemma.startswith('»'):
            out_lemma = ''.join(out_lemma[1:])
        if out_lemma.endswith('«') or out_lemma.endswith('»'):
            out_lemma = ''.join(out_lemma[:-1])
        if out_lemma.endswith('!') or out_lemma.endswith('?') or out_lemma.endswith(',') \
                or out_lemma.endswith('.'):
            out_lemma = ''.join(out_lemma[:-1])
    return out_lemma
  

def lemmatize(pipeline, text='Строка', keep_pos=True, keep_punct=False, min_token_size=4):
    entities = {'PROPN'}
    named = False
    memory = []
    mem_case = None
    mem_number = None
    tagged_propn = []

    # обрабатываем текст, получаем результат в формате conllu:
#     log(text)
    processed = pipeline.process(text)

    # пропускаем строки со служебной информацией:
    content = [l for l in processed.split('\n') if not l.startswith('#')]

    # извлекаем из обработанного текста леммы, тэги и морфологические характеристики
    tagged = [w.split('\t') for w in content if w]

    for t in tagged:
        if len(t) != 10:
            continue
            
        (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = t
        token = clean_token(token, misc)
        lemma = clean_lemma(lemma, pos)
        if pos != 'NUM' and len(token) < min_token_size:
            continue
        if not lemma or not token:
            continue
        if pos in entities:
            if '|' not in feats:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            morph = {el.split('=')[0]: el.split('=')[1] for el in feats.split('|')}
            if 'Case' not in morph or 'Number' not in morph:
                tagged_propn.append('%s_%s' % (lemma, pos))
                continue
            if not named:
                named = True
                mem_case = morph['Case']
                mem_number = morph['Number']
            if morph['Case'] == mem_case and morph['Number'] == mem_number:
                memory.append(lemma)
                if 'SpacesAfter=\\n' in misc or 'SpacesAfter=\s\\n' in misc:
                    named = False
                    past_lemma = '::'.join(memory)
                    memory = []
                    tagged_propn.append(past_lemma + '_PROPN ')
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))
        else:
            if not named:
                if pos == 'NUM' and token.isdigit():  # Заменяем числа на xxxxx той же длины
                    lemma = num_replace(token)
                tagged_propn.append('%s_%s' % (lemma, pos))
            else:
                named = False
                past_lemma = '::'.join(memory)
                memory = []
                tagged_propn.append(past_lemma + '_PROPN ')
                tagged_propn.append('%s_%s' % (lemma, pos))

    if not keep_punct:
        tagged_propn = [word for word in tagged_propn if word.split('_')[1] != 'PUNCT']
    if not keep_pos:
        tagged_propn = [word.split('_')[0] for word in tagged_propn]
    return tagged_propn
  
  
def tag_ud(text='Текст нужно передать функции в виде строки!'):
    log("Lemmatizing started...")
    udpipe_model_url = config['udpipe_model_url']
    udpipe_filename = udpipe_model_url.split('/')[-1]

    if not os.path.isfile(udpipe_filename):
        log('Lemmatizing: UDPipe model not found. Downloading...')
        wget.download(udpipe_model_url)

    log('Lemmatizing: loading the model...')
    model = Model.load(udpipe_filename)
    process_pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')

    log('Lemmatizing: processing input...')
    
    result = []
    for line in text:
        # line = unify_sym(line.strip()) # здесь могла бы быть ваша функция очистки текста
        output = lemmatize(process_pipeline, text=line)
        result.append(output)
  
    return result


def get_onehot(paragraph):

    paragraph_onehot = [np.zeros(config['one_hot_size'], dtype=np.int32)]

    for i in range(len(paragraph)):
        word = paragraph[i]
        # есть ли слово в модели? Может быть, и нет
        onehot = np.zeros(config['one_hot_size'], dtype=np.int32)
        if word in word2vec.vocab:
            word_embedding = word2vec[word]
            word_cluster = kmeans.predict([word_embedding])
            onehot.itemset(word_cluster[0], 1)
#             log(word)
#             log(word_cluster)
#             log(word2vec.similar_by_vector(kmeans.cluster_centers_[word_cluster[0]], 3))
#             log('\n')
        else:
            # Увы!
#             log('"' + word + ' is not present in the model')
#             log()
            pass
        paragraph_onehot.append(onehot)

    paragraph_onehot = np.sum(paragraph_onehot, axis=0)
    paragraph_onehot = paragraph_onehot.astype(np.int32)
    
    return paragraph_onehot


# get all paragraphs from charter, lemmatize words inside and get onehot vectors for each paragraph
def load_charters(path, file_names):

    log("Loading charters started...")

    pre_labels = []
    texts = []

    log("Extracting texts...")
    for filename in file_names:
        # log(filename[-3:])
        if filename[-3:] == 'ods':
            try:
                df = read_ods(path + "/" + filename, 1)
                df = df[~pd.isna(df[df.columns[0]])]
                df = df.astype({df.columns[0]: str})

                pre_labels.extend(df[df.columns[1]].values.tolist())
                texts.extend(df[df.columns[0]].values.tolist())
            except Exception as e:
                log("Error on file")
                log(e)
                log(filename)

        elif filename[-3:] == 'txt':
            try:
                with open(path + "/" + filename, encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    if line[-1:] == '\n':
                        line = line[:-1]
                    if len(line) > 0:
                        pre_labels.append(0)
                        texts.append(line)
            except Exception as e:
                log("Error on file")
                log(e)
                log(filename)

        else:
            raise Exception('Wrong files', 'Not supported file extension. Supported extensions are: "ods", "csv"')

    return pre_labels, texts


# get all paragraphs from charter, lemmatize words inside and get onehot vectors for each paragraph
def parse_charters(pre_labels, texts):

    log("Parsing charters (lemmatizing and one-hot vectors creating) started...")

    #

    log("Collecting labels...")
    last_know_value = -1
    bad_label_indexes = []
    num_classes = config['classifier']['num_classes']
    for i in range(len(pre_labels)):
        try:
            x = pre_labels[i]
            if x is None or np.isnan(x):
                if last_know_value == -1:
                    last_know_value = 0
                    log('WARNING: unknown start label from document:', texts[i])
                pre_labels[i] = last_know_value
                x = last_know_value
            if x >= num_classes:
                bad_label_indexes.insert(0, i)
                log('WARNING: bad class label:', x, texts[i])
            else:
                last_know_value = pre_labels[i]
        except Exception as e:
            log(e)

    for bad in bad_label_indexes:
        del pre_labels[bad]
        del texts[bad]

    lemmatized_paragraphs = tag_ud(text=texts)

    if config['word2vec']['bigrams'] == "true":
        # bi-grams
        log("Creating bigrams...")
        phrases = gensim.models.phrases.Phrases(lemmatized_paragraphs, min_count=20, threshold=7, progress_per=300)
        bigram = gensim.models.phrases.Phraser(phrases)
        lemmatized_paragraphs = bigram[lemmatized_paragraphs]

        if config['word2vec']['trigrams'] == "true":
            # tri-grams and four-grams
            log("Creating trigrams...")
            phrases = gensim.models.phrases.Phrases(lemmatized_paragraphs, min_count=20, threshold=4, progress_per=300)
            bigram = gensim.models.phrases.Phraser(phrases)
            lemmatized_paragraphs = bigram[lemmatized_paragraphs]

    # word_freq = defaultdict(int)
    # for sent in lemmatized_paragraphs:
    #     for i in sent:
    #         word_freq[i] += 1
    #
    # log(sorted(word_freq, key=word_freq.get, reverse=True)[:100])

    return lemmatized_paragraphs, pre_labels


def make_one_hots(lemmatized_paragraphs, pre_labels=None):

    processed_charter_onehots = []
    processed_labels = []
    processed_lemmatized_words = []

    log('Converting to one-hot vectors...')
    num_paragraphs_for_print = len(lemmatized_paragraphs) // 5 + 1
    for i in range(len(lemmatized_paragraphs)):
        if i % num_paragraphs_for_print == 0:
            log('Converting to one-hot vectors... {}/{}'.format(i+1, len(lemmatized_paragraphs)))
        paragraph = lemmatized_paragraphs[i]
        processed_lemmatized_words.extend(paragraph)
        if pre_labels is None:
            label = 0
        else:
            label = pre_labels[i]
        if label >= 0:
            processed_charter_onehots.append(get_onehot(paragraph))
            processed_labels.append(label)
        else:
            log("WARNING: not valid data:", label, paragraph)
    log('Converting to one-hot vectors... {}/{}'.format(i+1, len(lemmatized_paragraphs)))

    processed_labels = np.array(processed_labels)

    processed_charter_onehots = np.array(processed_charter_onehots, dtype=np.int32)
    processed_labels = processed_labels.astype(np.int32)

    log("Parsing charters (lemmatizing and one-hot vectors creating) finished")

    return processed_labels, processed_charter_onehots, processed_lemmatized_words


def find_n_grams(lemmatized_paragraphs):

    all_n_grams = []
    all_n_grams_str = []
    for paragraph in lemmatized_paragraphs:
        n_grams = []
        for gram in paragraph:
            if len([m.start() for m in re.finditer('_', gram)]) > 1:
                n_grams.append(gram)
        n_grams_unique = np.unique(np.array(n_grams)).tolist()
        all_n_grams.append(n_grams)
        all_n_grams_str.append(", ".join(n_grams_unique))

    return all_n_grams, all_n_grams_str


# get all files for evaluating, fill with predictions and save
def save_predictions(path, file_names, predictions):

    log("Filling files with predictions started...")

    classes = []
    probabilities = []
    for single_prediction in predictions:
        predicted_class = int(single_prediction['classes'][:1][0])
        classes.append(predicted_class)
        probability = single_prediction['probabilities'][predicted_class]
        probabilities.append(probability)

    classes = np.array(classes)
    probabilities = np.array(probabilities)

    last_filled_df_size = 0
    for filename in file_names:
        # log(filename[-3:])
        if filename[-3:] == 'ods':
            try:
                df = read_ods(path + "/" + filename, 1)
                df = df[~pd.isna(df[df.columns[0]])]
                df = df.astype({df.columns[0]: str})
            except Exception as e:
                log("Error on file")
                log(e)
                log(filename)

        elif filename[-3:] == 'txt':
            try:
                df = pd.DataFrame(np.array([]), columns=['texts'])
                with open(path + "/" + filename, encoding='utf-8') as f:
                    lines = f.readlines()

                texts = []
                for line in lines:
                    if line[-1:] == '\n':
                        line = line[:-1]
                    if len(line) > 0:
                        texts.append(line)

                df['texts'] = texts
            except Exception as e:
                log("Error on file")
                log(e)
                log(filename)

        else:
            raise Exception('Wrong files', 'Not supported file extension. Supported extensions are: "ods", "csv"')

        df['Predicted class'] = classes[last_filled_df_size:last_filled_df_size+len(df)]
        df['Probability'] = probabilities[last_filled_df_size:last_filled_df_size+len(df)]
        last_filled_df_size += len(df)

        # df = df.drop('label 1', axis=1)

        dir_name = os.path.dirname(filename)
        if not os.path.exists(data_root_path + "Predictions" + "/" + dir_name):
            os.mkdir(data_root_path + "Predictions" + "/" + dir_name)
        df.to_csv(data_root_path + "Predictions" + "/" + filename[:-3] + "csv")

    log("Filling files with predictions finished")

    return classes, probabilities


# get all files for testing, fill with predictions, check with exist labels and save
def save_test_results(path, file_names, predictions, true_classes, lemmatized_paragraphs):

    log("Filling test files with test predictions started...")

    # Clean Testing folder
    if os.path.exists(data_root_path + "Testing"):
        for the_file in os.listdir(data_root_path + "Testing"):
            file_path = os.path.join(data_root_path + "Testing", the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    shutil.rmtree(data_root_path + "Testing")
            except Exception as e:
                log(e)
    else:
        os.mkdir(data_root_path + "Testing")

    n_grams, n_grams_str = find_n_grams(lemmatized_paragraphs)

    predicted_classes = []
    probabilities = []
    for single_prediction in predictions:
        predicted_class = int(single_prediction['classes'][:1][0])
        predicted_classes.append(predicted_class)
        probability = single_prediction['probabilities'][predicted_class]
        probabilities.append(probability)

    predicted_classes = np.array(predicted_classes)
    probabilities = np.array(probabilities)

    n_gram_classes = {}
    num_classes = config['classifier']['num_classes']
    for label in range(num_classes):
        n_gram_classes[label] = {}

    for i in range(len(n_grams)):
        label = predicted_classes[i]
        labels_n_grams = n_grams[i]
        for n_gram in labels_n_grams:
            if n_gram in n_gram_classes[label]:
                n_gram_classes[label][n_gram] += 1
            else:
                n_gram_classes[label][n_gram] = 1

    for i in range(len(n_gram_classes)):
        n_gram_classes[i] = sorted(n_gram_classes[i].items(), key=lambda kv: kv[1], reverse=True)
        n_gram_classes[i] = {x[0]: x[1] for x in n_gram_classes[i]}

    # for label in range(len(n_gram_classes)):
    #     n_gram_classes[label] = np.unique(np.array(n_gram_classes[label])).tolist()

    with open(data_root_path + "Testing/n_gram_classes.json", 'w', encoding='utf8') as out_file:
        json.dump(n_gram_classes, out_file, ensure_ascii=False, indent=4)

    last_filled_df_size = 0
    for filename in file_names:
        #     log(filename)
        try:
            df = read_ods(path + "/" + filename, 1)
            df = df[~pd.isna(df[df.columns[0]])]
            df = df.astype({df.columns[0]: str})

            df[df.columns[1]] = true_classes[last_filled_df_size:last_filled_df_size+len(df)]
            df['Predicted class'] = predicted_classes[last_filled_df_size:last_filled_df_size+len(df)]
            df['Probability'] = probabilities[last_filled_df_size:last_filled_df_size+len(df)]
            df['Correct'] = predicted_classes[last_filled_df_size:last_filled_df_size+len(df)] == true_classes[last_filled_df_size:last_filled_df_size+len(df)]
            df['n_grams'] = n_grams_str[last_filled_df_size:last_filled_df_size+len(df)]
            last_filled_df_size += len(df)

            df.to_csv(data_root_path + "Testing" + "/" + filename[:-3] + "csv")
        except Exception as e:
            log("Error on file")
            log(e)
            log(filename)
            log("true_classes: {}, predicted_classes: {}, probabilities: {}, n_grams_str: {}, "
                  "lemmatized_paragraphs: {}, last_filled_df_size: {}".format(
                len(true_classes), len(predicted_classes), len(probabilities),
                len(n_grams_str), len(lemmatized_paragraphs), last_filled_df_size))

    accuracy_score = np.sum(true_classes == predicted_classes) / len(true_classes)
    log("\nTest Accuracy: {:0.2f}%\n".format(accuracy_score*100))

    log("Filling test files with test predictions finished")


############################################################################

# Pipeline

def load_word2vec_model():

    if START_STAGE >= 1:
        # Load word2vec model
        log("Loading word2vec model...")
        if config['word2vec']['python_trained'] == "true":
            model = gensim.models.KeyedVectors.load(config['word2vec']['saved_model'])
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(config['word2vec']['saved_model'], binary=True)
        log("word2vec model loaded")

    else:
        log("----------------- TRAIN WORD2VEC -----------------")

        cores = multiprocessing.cpu_count()

        if config['word2vec']['algorithm'] == "skipgram":
            use_skip_gram = 1
        elif config['word2vec']['algorithm'] == "cbow":
            use_skip_gram = 0
        else:
            raise Exception('Wrong config', 'word2vec.algorithm field from config can take "skipgram" or "cbow" values.')

        log("Creating word2vec model using " + config['word2vec']['algorithm'] + "...")
        model = gensim.models.Word2Vec(
            sg=use_skip_gram,
            min_count=20,
            window=2,
            size=300,
            sample=6e-5,
            alpha=0.03,
            min_alpha=0.0007,
            negative=20,
            workers=cores-1
        )

        train_and_test_path = data_root_path + "Train"
        train_and_test_file_names = [f for f in os.listdir(train_and_test_path) if os.path.isfile(os.path.join(train_and_test_path, f))]
        pre_labels, texts = load_charters(train_and_test_path, train_and_test_file_names)
        lemmatized_paragraphs, pre_labels = parse_charters(pre_labels, texts)

        t = time()
        model.build_vocab(lemmatized_paragraphs, progress_per=10000)
        log('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

        t = time()
        model.train(lemmatized_paragraphs, total_examples=model.corpus_count, epochs=30, report_delay=1)
        log('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

        model.init_sims(replace=True)

        model = model.wv

        model.save(config['word2vec']['saved_model'])

    return model


def clusterize_vocab():

    if START_STAGE >= 2:
        log("Loading KMeans model...")
        if not os.path.exists(saved_models_path + 'kmeans/kmeans.joblib'):
            raise Exception('Wrong start stage', 'KMeans model with clustered embeddings does not exist by path ' +
                            saved_models_path + 'kmeans/kmeans.joblib' +
                            ', please, add file or decrease start stage to "clusterize_vocab" or previous.')

        model = joblib.load(saved_models_path + 'kmeans/kmeans.joblib')

    else:
        log("----------------- CLUSTERIZING VOCAB EMBEDDINGS -----------------")
        vocab = []
        for word in word2vec.vocab.keys():
            vocab.append(word)

        vocab_embeddings = word2vec[vocab]

        log(str(len(vocab_embeddings)) + " vocab embeddings ready")

        model = KMeans(n_clusters=config['one_hot_size'], verbose=1)
        model.fit(vocab_embeddings)

        joblib.dump(model, saved_models_path + 'kmeans/kmeans.joblib')

    return model


def train():

    num_classes = config['classifier']['num_classes']

    # Specify feature
    feature_columns = [tf.feature_column.numeric_column("x", shape=[config['one_hot_size']], dtype=tf.dtypes.int32)]

    classifier_model = tf.estimator.DNNClassifier(
        hidden_units=[512, 256, 128],
        feature_columns=feature_columns,
        n_classes=num_classes,
        model_dir=saved_models_path + 'classifier/'
        #     activation_fn=tf.nn.tanh,
        #     optimizer=lambda: tf.train.AdamOptimizer(
        #         learning_rate=tf.train.exponential_decay(
        #             learning_rate=0.001,
        #             global_step=tf.train.get_global_step(),
        #             decay_steps=1000,
        #             decay_rate=0.96
        #         )
        #     )
    )

    train_and_test_path = data_root_path + "Train"
    train_and_test_file_names = [f for f in os.listdir(train_and_test_path) if os.path.isfile(os.path.join(train_and_test_path, f))]

    test_file_names = random.choices(train_and_test_file_names, k=int(0.2 * len(train_and_test_file_names)))
    if len(test_file_names) <= 0:
        test_file_names = random.choices(train_and_test_file_names)

    if START_STAGE >= 3:
        # Load model
        if len(os.listdir(saved_models_path + 'classifier')) == 0:
            raise Exception('Wrong start stage', 'Classifier model does not exist by path ' +
                            saved_models_path + 'classifier/' +
                            ', please, add files or decrease start stage to "train_classifier" or previous.')
    else:
        log("----------------- TRAIN CLASSIFIER -----------------")

        # Clean up classifier model folder (with checkpoints)
        for the_file in os.listdir(saved_models_path + 'classifier'):
            file_path = os.path.join(saved_models_path + 'classifier', the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                log(e)

        train_file_names = []
        for f in train_and_test_file_names:
            if not f in test_file_names:
                train_file_names.append(f)

        log("Train files num: {}, test files num: {}, total num: {}".format(
            len(train_file_names), len(test_file_names), len(train_and_test_file_names)))

        pre_labels, texts = load_charters(train_and_test_path, train_file_names)
        lemmatized_paragraphs, pre_labels = parse_charters(pre_labels, texts)
        labels, charter_onehots, _ = make_one_hots(lemmatized_paragraphs, pre_labels)

        if int(np.max(labels)) + 1 > num_classes:
            debug_classes = {}
            for label in range(int(np.max(labels)) + 1):
                debug_classes[label] = 0

            for label in labels:
                debug_classes[label] += 1

            with open(data_root_path + "debug_classes.json", 'w', encoding='utf8') as out_file:
                json.dump(debug_classes, out_file, ensure_ascii=False, indent=4)

            raise Exception('Wrong classifier model', 'Classifier model should be trained on more number of classes'
                                                      ', exist: {}, from config: {}. '
                                                      'Please, increase "classifier.num_classes" field in the '
                                                      'config file.'.format(int(np.max(labels)) + 1, num_classes))

        X_train = charter_onehots
        y_train = labels

        # Define the training inputs
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_train},
            y=y_train.astype(np.int32),
            num_epochs=None,
            batch_size=config['classifier']['batch_size'],
            shuffle=True
        )

        classifier_model.train(input_fn=train_input_fn, steps=config['classifier']['steps'])

    test(classifier_model, train_and_test_path, test_file_names)

    return classifier_model


def test(classifier_model, train_and_test_path, test_file_names):

    if START_STAGE >= 4:
        pass
    else:
        log("----------------- TESTING CLASSIFIER -----------------")

        test_file_names.sort()
        pre_labels, texts = load_charters(train_and_test_path, test_file_names)
        lemmatized_paragraphs, pre_labels = parse_charters(pre_labels, texts)
        labels, charter_onehots, _ = make_one_hots(lemmatized_paragraphs, pre_labels)

        X_test = charter_onehots
        y_test = labels

        # Define the test inputs
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": X_test},
            # y=y_test.astype(np.int32),
            num_epochs=1,
            shuffle=False
        )

        testing_result = classifier_model.predict(input_fn=test_input_fn)

        save_test_results(train_and_test_path, test_file_names, testing_result, y_test, lemmatized_paragraphs)


def predict(classifier_model, texts=None):

    all_classes = []
    all_probabilities = []

    if START_STAGE >= 5:
        pass
    else:
        log("----------------- PREDICTION FROM CLASSIFIER -----------------")

        # Clean Predictions folder
        if os.path.exists(data_root_path + "Predictions"):
            for the_file in os.listdir(data_root_path + "Predictions"):
                file_path = os.path.join(data_root_path + "Predictions", the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    else:
                        shutil.rmtree(file_path)
                except Exception as e:
                    log(e)
        else:
            os.mkdir(data_root_path + "Predictions")

        if texts is None:
            texts = []

            # Take files from Evaluate
            prediction_path = data_root_path + "Evaluate"

            all_prediction_file_names = []
            for dir in os.listdir(prediction_path):
                all_prediction_file_names.extend([dir + "/" + f for f in os.listdir(prediction_path + "/" + dir) if os.path.isfile(os.path.join(prediction_path + "/" + dir, f))])
            all_prediction_file_names.sort()

            predict_batch_size = 500
            predict_batch_num = int(len(all_prediction_file_names) / predict_batch_size) + 1
            for i in range(predict_batch_num):

                log("Batch {} from {}".format(i+1, predict_batch_num))

                prediction_file_names = all_prediction_file_names[i * predict_batch_size:(i + 1) * predict_batch_size]

                pre_labels, batched_texts = load_charters(prediction_path, prediction_file_names)
                lemmatized_paragraphs, _ = parse_charters(pre_labels, batched_texts)
                _, predict_charter_one_hots, _ = make_one_hots(lemmatized_paragraphs)

                # Define the predict inputs
                predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": predict_charter_one_hots},
                    num_epochs=1,
                    shuffle=False
                )

                predictions_result = classifier_model.predict(input_fn=predict_input_fn)

                classes, probabilities = save_predictions(prediction_path, prediction_file_names, predictions_result)

                all_classes.extend(np.array(classes))
                all_probabilities.extend(np.array(probabilities))
                texts.extend(batched_texts)

        else:

            predict_batch_size = 500
            predict_batch_num = int(len(texts) / predict_batch_size) + 1
            for i in range(predict_batch_num):

                log("Batch {} from {}".format(i+1, predict_batch_num))

                pre_labels = []
                batched_texts = texts[i * predict_batch_size:(i + 1) * predict_batch_size]
                lemmatized_paragraphs, _ = parse_charters(pre_labels, batched_texts)
                _, predict_charter_one_hots, _ = make_one_hots(lemmatized_paragraphs)

                # Define the predict inputs
                predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": predict_charter_one_hots},
                    num_epochs=1,
                    shuffle=False
                )

                predictions_result = classifier_model.predict(input_fn=predict_input_fn)

                classes = []
                probabilities = []
                for single_prediction in predictions_result:
                    predicted_class = int(single_prediction['classes'][:1][0])
                    classes.append(predicted_class)
                    probability = single_prediction['probabilities'][predicted_class]
                    probabilities.append(probability)

                all_classes.extend(np.array(classes))
                all_probabilities.extend(np.array(probabilities))

    return all_classes, all_probabilities, texts


########################################################################################


def init(config_, saved_models_path_, start_stage_):

    global config
    global data_root_path
    global saved_models_path
    global START_STAGE

    global word2vec
    global kmeans
    global dnn_classifier_model

    with open(config_) as json_file:
        config = json.load(json_file)

    data_root_path = config['data_root_path'] + "/"
    saved_models_path = saved_models_path_

    logging.basicConfig(filename=data_root_path+"logs.log",
                        format=u"%(levelname)s - %(asctime)s: %(message)s",
                        datefmt='%H:%M:%S', filemode="w", level=logging.INFO)

    log("Tensorflow version:", tf.version.VERSION)

    START_STAGE = start_stage_

    word2vec = load_word2vec_model()

    kmeans = clusterize_vocab()

    dnn_classifier_model = train()


def get_markup(texts=None):
    if dnn_classifier_model is None:
        init("default.config", "saved_models/", 4)

    classes, probabilities, texts = predict(dnn_classifier_model, texts)

    result = {}
    for i in range(len(classes)):
        if not classes[i] in result:
            result[int(classes[i])] = []
        result[int(classes[i])].append((texts[i], round((probabilities[i]), 3)))

    return result
