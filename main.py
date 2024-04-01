import os
import pickle

import click
import matplotlib.pyplot as plt

import keras
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

from utilities.utils import *

@click.group(chain=False, invoke_without_command=True)
@click.option('--debug/--no-debug', help='debug mode flag', default=False)
@click.pass_context
def router_cmd(ctx: click.Context, debug):
    ctx.obj['debug_mode'] = debug 
    invoked_subcommand = ctx.invoked_subcommand 
    if invoked_subcommand is None:
        logger.debug('no subcommand were called')
    else:
        logger.debug(f'{invoked_subcommand} was called')

@router_cmd.command()
@click.option('--path2destination', help='where the dataset will be saved', type=str, default='data/')
def grabber(path2destination):
    if not os.path.exists(path2destination):
        os.makedirs(path2destination)
    
    max_features = 20000
    
    train_data, test_data = imdb.load_data(num_words=max_features) 
    with open(os.path.join(path2destination, 'imdb_train.pkl'), 'wb') as fp:
        pickle.dump(train_data, fp)
    with open(os.path.join(path2destination, 'imdb_test.pkl'), 'wb') as fp:
        pickle.dump(test_data, fp)
    
    logger.info('Dataset successfully saved ...!')

@router_cmd.command()
@click.option('--path2data', help='path to dataset', type=click.Path(True), default='data/imdb_train.pkl')
@click.option('--path2models', help='models storage', type=str, default='models/')
@click.option('--path2metrics', help='metrics storage', type=str, default='metrics/')
@click.option('--epochs', help='number of epochs', type=int, default=10)
@click.option('--batch_size', help='batch size', type=int, default=32)
def learn(path2data, path2models, batch_size, epochs, path2metrics):
    if not os.path.exists(path2models):
        os.makedirs(path2models)
    
    max_features = 20000
    maxlen = 80

    with open(path2data, 'rb') as fp:
        train_data = pickle.load(fp)
    X_train, y_train = train_data
    X_train = pad_sequences(X_train, maxlen=maxlen)
    
    model = keras.Sequential([
        keras.layers.Embedding(max_features, 128),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    plt.plot(history.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Losses curve'])
    plt.savefig(os.path.join(path2metrics, 'loss.png'))
    
    model.save(os.path.join(path2models, 'model.h5'))
    logger.info('The model was saved ...!')

@router_cmd.command()
@click.option('--path2data', help='path to dataset', type=click.Path(True), default='data/imdb_test.pkl')
@click.option('--path2model', help='path to model', type=click.Path(True), default='models/model.h5')
def inference(path2data, path2model):
    logger.debug('Inference...')
    
    maxlen = 80
    with open(path2data, 'rb') as fp:
        test_data = pickle.load(fp)
    X_test, y_test = test_data
    X_test = pad_sequences(X_test, maxlen=maxlen)
    print(decode_review(X_test, 0))
        
    model = keras.models.load_model(path2model)
    
    predictions = model.predict(X_test)
    pred_labels = []
    for i in predictions:
        if i >= 0.5:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

if __name__ == '__main__':
    router_cmd(obj={})