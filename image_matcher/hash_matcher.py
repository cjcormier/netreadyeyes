import imagehash as ih
import numpy as np
from PIL import Image


def find_minimum_hash_difference(image, hash_pool, hash_size=32):
    image_object = Image.fromarray(image.astype('uint8'), 'RGB')
    card_hash = _create_and_flatten_perceptual_hash(image_object, hash_size)
    hash_pool['diff'] = hash_pool['card_hash_%d' % hash_size]
    hash_pool['diff'] = hash_pool['diff'].apply(lambda x: np.count_nonzero(x != card_hash))
    return hash_pool[hash_pool['diff'] == min(hash_pool['diff'])].iloc[0], \
           min(hash_pool['diff'])


def _create_and_flatten_perceptual_hash(image, hash_size):
    return ih.phash(image, hash_size=hash_size).hash.flatten()


def flatten_hash_array(card_pool):
    card_hk = 'card_hash_32'
    card_pool = card_pool[
        ['id', 'name', 'set', 'collector_number', card_hk]
    ]
    card_pool[card_hk] = card_pool[card_hk].apply(lambda x: x.hash.flatten())
    return card_pool