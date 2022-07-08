import os

import jsonpickle

from model.block import Block
from ops.block_state_builder import BlockStateBuilder
from utils.draw_architecture import DrawArchitecture


def draw_architectures():
    architectures = []
    for filename in os.listdir('architectures'):
        if filename.endswith('.json'):
            f = open(f'architectures/{filename}', 'r')
            json_str = f.read()
            architecture = jsonpickle.decode(json_str)
            f.close()
            architectures.append(architecture)

    draw = DrawArchitecture('')
    draw.draw_architectures(architectures)


def test_dag(key):
    f = open(f'./restore/architectures/{key}.json', 'r')
    json_str = f.read()
    architecture = jsonpickle.decode(json_str)
    f.close()
    draw = DrawArchitecture('')
    draw.test_dag(architecture)

def draw_circular_ref():
    architecture = BlockStateBuilder.get_basic_architecture()

    architecture.add_block(Block(['x'], 'x_lin_f', activation='linear_b'))
    architecture.add_block(Block(['h'], 'h_lin_f', activation='linear_b'))
    architecture.add_block(Block(['x_lin_f', 'f_th'], 'f_add_1', combination='add'))
    architecture.add_block(Block(['f_add_1', 'h_lin_f'], 'f_add', combination='add'))
    architecture.add_block(Block(['f_add'], 'f', activation='sigmoid'))

    architecture.add_block(Block(['f_add'], 'f_th', activation='tanh'))
    architecture.add_block(Block(['f', 'f_th'], 'c_add_1', combination='add'))

    # architecture.add_block(Block(['f_add', 'x_lin_f'], 'c_add', combination='add'))

    # architecture.add_block(Block(['c_add', 'c_add_1'], 'bf_add', combination='add'))

    architecture.blocks['h_next'].inputs = ['c_add_1']

    architecture.blocks.pop('c_1')
    architecture.blocks.pop('c_next')
    architecture.blocks.pop('x_lin_r_actv')
    architecture.blocks.pop('x_lin_r')
    architecture.blocks.pop('h_lin_r')
    architecture.blocks.pop('x_lin_r_add')

    draw = DrawArchitecture('circ_ref')
    draw.draw_architectures([architecture])

if __name__ == '__main__':

    # draw_architectures()

    # test_dag('rdm51_1')

    draw_circular_ref()
