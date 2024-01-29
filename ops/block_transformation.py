import copy
import os.path
import string

import jsonpickle

from config.env_config import EnvironmentConfig
from model.architecture import Architecture
from model.block import Block
from ops.probability_distribution import ProbabilityDistribution
from utils.logger import LOG
from utils.random_generator import RandomGenerator

LOG = LOG.get_instance().get_logger()

REMOVE_TRANSFORMATIONS = ["remove_unit", "remove_connection"]

"""
This class is responsible for the network morphism (performing network transformations).
"""

class AddUnitTransformation(object):
    def __init__(self, input_block_key, output_key, activation, receiving_block_key):
        self.input_block_key = input_block_key
        self.output_key = output_key
        self.activation = activation
        self.receiving_block_key = receiving_block_key
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Added {self.activation} unit from {self.input_block_key} to {self.receiving_block_key}, new unit key={self.output_key}.'


class RemoveUnitTransformation(object):
    def __init__(self, block_to_remove):
        self.block_to_remove = block_to_remove
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Removed unit {self.block_to_remove}.'


class AddConnectionTransformation(object):
    def __init__(self, receiving_block_key, new_receiving_key, input_block_key, force_recurrent):
        self.receiving_block_key = receiving_block_key
        self.new_receiving_key = new_receiving_key
        self.input_block_key = input_block_key
        self.force_recurrent = force_recurrent
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Added {"recurrent" if self.force_recurrent else ""} connection from {self.input_block_key} to {self.receiving_block_key}, updated new key={self.new_receiving_key}.'


class RemoveConnectionTransformation(object):
    def __init__(self, random_block_key, new_output):
        self.random_block_key = random_block_key
        self.new_output = new_output
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Removed connection from {self.random_block_key}, updated unit key={self.new_output}.'


class ChangeRandomActivationTransformation(object):
    def __init__(self, random_block_key, activation, original_activation):
        self.random_block_key = random_block_key
        self.activation = activation
        self.original_activation = original_activation
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Changed {self.original_activation} activation to {self.activation} for {self.random_block_key}.'


class ChangeCombinationConnectionTransformation(object):
    def __init__(self, random_block_key, new_combination, block_combination):
        self.random_block_key = random_block_key
        self.new_combination = new_combination
        self.block_combination = block_combination
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Changed {self.block_combination} combination to {self.new_combination} for {self.random_block_key}.'


class FlipNetworkTypeTransformation(object):
    def __init__(self, new_type):
        self.new_type = new_type
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        previous = 'Jordan' if self.new_type else 'Elman'
        return f'Flipped network type from {previous} to {"Elman" if self.new_type else "Jordan"}.'


class DifferentiableCombineActivateGateTransformation(object):
    def __init__(self, inputs, input_activations, input_combination, final_activation, output_key, receiving_block_key,
                 inp_1_key, inp_2_key, comb_key, new_receiving_key, merge_combination):
        self.inputs = inputs
        self.input_activations = input_activations
        self.input_combination = input_combination
        self.final_activation = final_activation
        self.output_key = output_key
        self.receiving_block_key = receiving_block_key
        self.inp_1_key = inp_1_key
        self.inp_2_key = inp_2_key
        self.comb_key = comb_key
        self.new_receiving_key = new_receiving_key
        self.merge_combination = merge_combination
        self.hash = hash(self)
        self.architecture_identifier = None

    def get_transformation_as_string(self):
        return f'Added differentiable combine activation transformation {self.output_key} with {self.final_activation} and {self.input_combination}' \
               f' to {self.receiving_block_key}, output key is={self.output_key}.'


class CombineActivateGateTransformation(object):
    def __init__(self, inputs, input_combinations, activations, output_key, receiving_block_key):
        self.inputs = inputs
        self.input_combinations = input_combinations
        self.activations = activations
        self.output_key = output_key
        self.receiving_block_key = receiving_block_key
        self.hash = hash(self)
        self.architecture_identifier = None


class ListOfTransformations(object):
    def __init__(self):
        self.transformations = []

    def get_hash(self):
        return hash(self)


class BlockTransformation:
    __instance = None

    def __init__(self, generation=0):
        self.exclude_linear_layer_actions = True
        self.generation = generation
        self.transformation_ids = {
            'add_unit': 'au',
            'remove_unit': 'ru',
            'add_gate': 'ag',
            'remove_gate': 'rg',
            'add_connection': 'ac',
            'remove_connection': 'rc',
            'add_recurrent_connection': 'arc',
            'remove_recurrent_connection': 'rrc',
            'change_random_activation': 'cra',
            'change_combination_connection': 'ccc',
            'new_gate_input': 'ngi',
            'linear_block': 'lnbl',
            'output_layer': 'outpt_lay'
        }
        self.transformations = {}
        self.linear_counts = 0
        self.non_linear_counts = 0
        self.gates_added = 0
        self.x_linear_connection = 0
        BlockTransformation.__instance = self

    @staticmethod
    def get_instance():
        if BlockTransformation.__instance is None:
            BlockTransformation()
            return BlockTransformation.__instance
        return BlockTransformation.__instance

    def transform_architecture(self, architecture: Architecture, transformation_count=None, initial_generation=False):
        transformation_hash = self.perform_transformation(architecture, transformation_count=transformation_count, initial_generation=initial_generation)
        for key in architecture.blocks.keys():
            architecture.blocks[key].input_dimension = None
            architecture.blocks[key].output_dimension = None
            if len(architecture.blocks[key].activation) > 0:
                architecture.blocks[key].activation = [architecture.blocks[key].activation[0]]

        architecture.update_generation()
        return transformation_hash

    def perform_transformation(self, architecture: Architecture, transformation_count=None, initial_generation=False):

        transformation_list = ListOfTransformations()
        if transformation_count is None:
            if self.generation == 0:
                number_of_transformations = RandomGenerator.randint(EnvironmentConfig.get_config('initial_population_transformations_min'),
                                  EnvironmentConfig.get_config('initial_population_transformations')+1)
            else:
                number_of_transformations = ProbabilityDistribution.get_number_of_transformations()
        else:
            number_of_transformations = RandomGenerator.randint(1, transformation_count + 1)

        if EnvironmentConfig.get_config('override_transformation_count') > 0 and self.generation > 0:
            number_of_transformations = EnvironmentConfig.get_config('override_transformation_count')
            if number_of_transformations > 1:
                number_of_transformations = RandomGenerator.randint(1, number_of_transformations + 1)

            LOG.info(f'{"=" * 10} [OVERRIDE] {number_of_transformations} transformations {"=" * 10}')
        else:
            LOG.info(f'{"=" * 10} {number_of_transformations} transformations {"=" * 10}')

        options = copy.deepcopy(EnvironmentConfig.get_config('network_transformations'))

        gate_transformation = False
        if 'add_gate' in options and EnvironmentConfig.get_config('gate_unit_option') == 'limit_morphism' and  self.generation > 0 and not initial_generation:
            options.remove('add_gate')
            gate_transformation = RandomGenerator.randint(11) < 3
            if gate_transformation:
                number_of_transformations = 1
                LOG.info('Forcing a single gate unit transformation.')

        transformations_performed = []
        for i in range(number_of_transformations):
            if self.generation == 0:
                # Don't allow removal transformations during initial population generation
                for _opt in REMOVE_TRANSFORMATIONS:
                    if _opt in options:
                        options.remove(_opt)

            distribution = ProbabilityDistribution.get_distribution(options)
            transformation = list(distribution.keys())[0]

            if gate_transformation:
                transformation = 'add_gate'

            LOG.info(f'Chosen transformation: {transformation}')
            if transformation == 'add_unit':
                new_transformation = self.generate_add_single_unit(architecture)
                self.add_single_unit(architecture, new_transformation)
            elif transformation == 'remove_unit':
                new_transformation = self.generate_remove_singe_unit(architecture)
                self.remove_single_unit(architecture, new_transformation)
            elif transformation == 'add_connection':
                new_transformation = self.generate_add_connection(architecture)
                self.add_connection(architecture, new_transformation)
            elif transformation == 'remove_connection':
                new_transformation = self.generate_remove_connection(architecture)
                self.remove_connection(architecture, new_transformation)
            elif transformation == 'change_random_activation':
                new_transformation = self.generate_change_random_activation(architecture)
                self.change_random_activation(architecture, new_transformation)
            elif transformation == 'add_recurrent_connection':
                new_transformation = self.generate_add_connection(architecture, force_recurrent=True)
                self.add_connection(architecture, new_transformation)
            elif transformation == 'change_combination_connection':
                new_transformation = self.generate_change_combination_connection(architecture)
                self.change_combination_connection(architecture, new_transformation)
            elif transformation == 'add_gate':
                new_transformation = self.generate_add_gate_unit(architecture)
                if type(new_transformation) is DifferentiableCombineActivateGateTransformation:
                    self.add_diff_comb_act_gate(architecture, new_transformation)
            elif transformation == 'flip_network_type':
                new_type = not architecture.elman_network
                new_transformation = FlipNetworkTypeTransformation(new_type)
                self.flip_network_type(architecture, new_transformation)
            else:
                raise Exception(f'Unknown transformation {transformation}.')

            new_transformation.architecture_identifier = architecture.identifier
            self.transformations[new_transformation.hash] = new_transformation
            architecture.transformation_history.append(new_transformation.hash)
            transformation_hash = new_transformation.hash
            transformation_list.transformations.append(transformation_hash)
            transformations_performed.append(new_transformation)

        self.persist_transformation_list(transformations_performed)

        if transformation_list.get_hash() in self.transformations.keys():
            raise Exception('Duplicate transformation hash.')

        self.transformations[transformation_list.get_hash()] = transformation_list
        self.update_architecture_output_layer_blocks(architecture)

        return transformation_list.get_hash()

    def update_architecture_output_layer_blocks(self, architecture: Architecture):
        for key in architecture.output_layer_keys:
            if architecture.blocks[key].combination:
                output_block = architecture.blocks[key]
                new_block_id = self.get_output_key(architecture, f'{key}_{output_block.combination}')
                new_block = Block(output_block.inputs, new_block_id, combination=output_block.combination)
                output_block.inputs = [new_block_id]
                output_block.combination = None
                architecture.add_block(new_block)

    def apply_transformation(self, architecture: Architecture, transformation_hash):
        transformation_list = self.transformations[transformation_hash]
        for transformation in transformation_list.transformations:
            self.apply_single_transformation(architecture, transformation)

    def apply_single_transformation(self, architecture: Architecture, transformation_hash):
        if transformation_hash not in self.transformations.keys():
            raise Exception(f'Unknown transformation {transformation_hash}.')

        if transformation_hash in architecture.transformation_history:
            print(f'Transformation already applied to state.')
            return

        transformation = self.transformations[transformation_hash]
        if type(transformation) is AddUnitTransformation:
            self.add_single_unit(architecture, transformation)
        elif type(transformation) is RemoveUnitTransformation:
            self.remove_single_unit(architecture, transformation)
        elif type(transformation) is AddConnectionTransformation:
            self.add_connection(architecture, transformation)
        elif type(transformation) is RemoveConnectionTransformation:
            self.remove_connection(architecture, transformation)
        elif type(transformation) is ChangeRandomActivationTransformation:
            self.change_random_activation(architecture, transformation)
        elif type(transformation) is ChangeCombinationConnectionTransformation:
            self.change_combination_connection(architecture, transformation)
        elif type(transformation) is DifferentiableCombineActivateGateTransformation:
            self.add_diff_comb_act_gate(architecture, transformation)
        else:
            raise Exception(f'Unknown transformation type {type(transformation)}.')

        architecture.transformation_history.append(transformation_hash)

    def add_single_unit(self, architecture: Architecture, _transformation: AddUnitTransformation):
        if _transformation.activation is None and _transformation.receiving_block_key is None:
            return

        transformation = copy.deepcopy(_transformation)
        input_block_key = transformation.input_block_key
        output_key = transformation.output_key
        activation = transformation.activation
        receiving_block_key = transformation.receiving_block_key

        if activation not in ['linear', 'linear_b']:
            """
            Allow for optionally performing a (weighted) linear activation
            """
            chance = str(RandomGenerator.choice(['lin', 'non']))
            if chance == 'lin':
                linear_activation = str(RandomGenerator.choice(['linear', 'linear_b']))
                new_lin_key = self.get_output_key(architecture, 'linear_block')
                new_linear_unit = Block([input_block_key], new_lin_key, activation=linear_activation)
                architecture.add_block(new_linear_unit)
                if new_lin_key == input_block_key:
                    print('')
                input_block_key = new_lin_key
                self.linear_counts += 1
            else:
                self.non_linear_counts += 1

        new_unit = Block([input_block_key], output_key, activation=activation)
        architecture.add_block(new_unit)

        receiving_block = architecture.blocks[receiving_block_key]
        for i, inp in enumerate(receiving_block.inputs):
            if inp == transformation.input_block_key:
                receiving_block.inputs[i] = output_key

        if output_key in architecture.blocks[receiving_block_key].inputs:
            LOG.info(
                f'Added a {activation} layer-unit between {input_block_key} and {receiving_block_key}, layer id={output_key}.')
            architecture.verify_blocks()
            architecture.blocks[output_key].transformations.append(transformation)
            architecture.blocks[receiving_block_key].transformations.append(transformation)
        else:
            raise Exception(
                f'Unable to add a {activation} layer-unit between {input_block_key} and {receiving_block_key}, layer id={output_key}.')

    def remove_single_unit(self, architecture: Architecture, _transformation: RemoveUnitTransformation):
        if _transformation.block_to_remove is None:
            return

        transformation = copy.deepcopy(_transformation)
        random_block_key = transformation.block_to_remove

        removed_activation = copy.deepcopy(architecture.blocks[random_block_key].activation[0])
        architecture.blocks[random_block_key].activation = []

        removed_block_uses = architecture.get_blocks_that_use(random_block_key)
        for arch in removed_block_uses:
            for i, inp_key in enumerate(architecture.blocks[arch].inputs):
                if inp_key == random_block_key:
                    architecture.blocks[arch].inputs[i] = architecture.blocks[random_block_key].inputs[0]

        architecture.blocks.pop(random_block_key)

        if random_block_key not in architecture.blocks.keys():
            architecture.verify_blocks()
            architecture.removed_blocks.append(random_block_key)
            LOG.info(f'Removed {removed_activation} layer from {random_block_key}.')
        else:
            raise Exception(f'Unable to remove {removed_activation} layer from {random_block_key}.')

    def add_connection(self, architecture: Architecture, _transformation: AddConnectionTransformation):
        if _transformation.receiving_block_key is None:
            return

        transformation = copy.deepcopy(_transformation)
        receiving_block_key = transformation.receiving_block_key
        new_receiving_key = transformation.new_receiving_key
        input_block_key = transformation.input_block_key
        force_recurrent = transformation.force_recurrent

        if input_block_key == 'x':
            linear_block_key = self.get_output_key(architecture, 'linear_block')
            linear_activation = RandomGenerator.choice(['linear', 'linear_b'])
            linear_block = Block([input_block_key], linear_block_key, activation=linear_activation, immutable=True)
            architecture.add_block(linear_block)
            input_block_key = linear_block_key
            self.x_linear_connection += 1

        architecture.blocks[receiving_block_key].identifier = new_receiving_key
        architecture.blocks[new_receiving_key] = architecture.blocks.pop(receiving_block_key)
        connection_block = Block([input_block_key, new_receiving_key], receiving_block_key, combination='add')
        architecture.add_block(connection_block)

        if new_receiving_key in architecture.blocks.keys() and receiving_block_key in architecture.blocks.keys() \
                and new_receiving_key in architecture.blocks[receiving_block_key].inputs \
                and input_block_key in architecture.blocks[receiving_block_key].inputs:
            architecture.verify_blocks()
            LOG.info(
                f'Added a {"recurrent" if force_recurrent else ""} connection between block {input_block_key} and {receiving_block_key}.')
            architecture.blocks[new_receiving_key].transformations.append(transformation)
            architecture.blocks[receiving_block_key].transformations.append(transformation)
            architecture.blocks[input_block_key].transformations.append(transformation)
        else:
            raise Exception(
                f'Unable to add a {"recurrent" if force_recurrent else ""} connection between block {input_block_key} and {receiving_block_key}.')

    def remove_connection(self, architecture: Architecture, _transformation: RemoveConnectionTransformation):
        if _transformation.random_block_key is None:
            return

        transformation = copy.deepcopy(_transformation)
        random_block_key = transformation.random_block_key
        new_output = transformation.new_output

        other_input = ''
        for i in architecture.blocks[random_block_key].inputs:
            if i != new_output:
                other_input = i

        blocks_that_uses_other_input = set()
        check_block = ''
        for block_key in architecture.blocks.keys():
            block = architecture.blocks[block_key]
            for e, inp in enumerate(block.inputs):
                if inp == other_input:
                    blocks_that_uses_other_input.add(copy.copy(block.identifier))

                if inp == random_block_key:
                    architecture.blocks[block_key].inputs[e] = new_output
                    check_block = block_key

        architecture.blocks.pop(random_block_key)

        if other_input not in ['x', 'h', 'c'] and len(
                blocks_that_uses_other_input) == 0 and other_input in architecture.blocks.keys():
            architecture.blocks.pop(other_input)

        if check_block in architecture.blocks.keys() and new_output in architecture.blocks[check_block].inputs:
            architecture.verify_blocks()
            LOG.info(
                f'Removed connection from {new_output} and {other_input} to {random_block_key}. New key is now {new_output}.')
            architecture.blocks[new_output].transformations.append(transformation)
            architecture.removed_blocks.append(random_block_key)
            architecture.transformation_history.append(transformation)
        else:
            raise Exception(f'Unable to remove connection from {new_output} and {other_input} to {random_block_key}.')

    def change_random_activation(self, architecture: Architecture,
                                 _transformation: ChangeRandomActivationTransformation):
        if _transformation.random_block_key is None:
            return

        transformation = copy.deepcopy(_transformation)
        random_block_key = transformation.random_block_key
        activation = transformation.activation
        original_activation = transformation.original_activation

        architecture.blocks[random_block_key].activation = [activation]
        if architecture.blocks[random_block_key].activation[0] == activation:
            architecture.verify_blocks()
            LOG.info(f'Changed block {random_block_key} activation layer from {original_activation} to {activation}.')
            architecture.blocks[random_block_key].transformations.append(transformation)
        else:
            raise Exception(
                f'Unable to change block {random_block_key} activation layer from {original_activation} to {activation}.')

    def change_combination_connection(self, architecture: Architecture,
                                      _transformation: ChangeCombinationConnectionTransformation):
        if _transformation.random_block_key is None:
            return

        transformation = copy.deepcopy(_transformation)
        random_block_key = transformation.random_block_key
        new_combination = transformation.new_combination
        block_combination = transformation.block_combination

        architecture.blocks[random_block_key].combination = new_combination

        if architecture.blocks[random_block_key].combination == new_combination:
            architecture.verify_blocks()
            LOG.info(f'Changed block {random_block_key} combination from {block_combination} to {new_combination}.')
            architecture.blocks[random_block_key].transformations.append(transformation)
        else:
            raise Exception(
                f'Unable to change block {random_block_key} combination from {block_combination} to {new_combination}.')

    def add_diff_comb_act_gate(self, architecture: Architecture,
                               _transformation: DifferentiableCombineActivateGateTransformation):
        if _transformation.receiving_block_key is None:
            return

        self.gates_added += 1

        transformation = copy.deepcopy(_transformation)

        input_block_2 = Block([transformation.inputs[1]], transformation.inp_2_key, activation=transformation.input_activations[1])

        if transformation.input_combination == 'sub' and transformation.inputs[0] == '1':
            # This allows for the inclusion of the GRU z-gate (1-z)
            combination_block = Block([1, input_block_2.identifier], transformation.comb_key, combination='sub')
        else:
            input_block_1 = Block([transformation.inputs[0]], transformation.inp_1_key, activation=transformation.input_activations[0])
            combination_block = Block([input_block_1.identifier, input_block_2.identifier], transformation.comb_key, combination=transformation.input_combination)
            architecture.add_block(input_block_1)

        architecture.add_block(input_block_2)
        architecture.add_block(combination_block)

        final_activation_block = Block([transformation.comb_key], transformation.output_key, activation=transformation.final_activation)
        architecture.add_block(final_activation_block)

        receiving_block_key = transformation.receiving_block_key
        new_receiving_key = transformation.new_receiving_key

        architecture.blocks[receiving_block_key].identifier = new_receiving_key
        architecture.blocks[new_receiving_key] = architecture.blocks.pop(receiving_block_key)

        connection_block = Block([transformation.output_key, new_receiving_key], receiving_block_key,
                                 combination=transformation.merge_combination)
        architecture.add_block(connection_block)

        if transformation.output_key in architecture.blocks.keys() and new_receiving_key in architecture.blocks.keys():
            architecture.verify_blocks()
            LOG.info(f'Added new differentiable gate {transformation.output_key} to {receiving_block_key}.')
            architecture.blocks[transformation.output_key].transformations.append(transformation)
            architecture.blocks[transformation.new_receiving_key].transformations.append(transformation)
            architecture.blocks[transformation.receiving_block_key].transformations.append(transformation)
        else:
            raise Exception(
                f'Unable to add differentiable gate {transformation.output_key} to {receiving_block_key}.')

    def flip_network_type(self, architecture: Architecture, _transformation: FlipNetworkTypeTransformation):
        """
            Not implemented for the study
        """
        print('Flip network type')

    def get_all_activation_layer_blocks(self, architecture: Architecture, exclude_outputs=True):
        blocks = set()
        exclusions = []
        if self.exclude_linear_layer_actions:
            exclusions = ['linear', 'linear_b']
        for key in architecture.blocks.keys():
            block = architecture.blocks[key]
            if len(block.activation) > 0 and block.activation[0] not in exclusions and not block.immutable:
                blocks.add(block.identifier)

        lst = list(blocks)
        static_layers = ['x', 'h', 'c']
        if exclude_outputs:
            static_layers.append('h_next')
            static_layers.append('c_next')

        for i in static_layers:
            if i in lst:
                lst.remove(i)
        return lst

    def get_all_connected_blocks(self, architecture: Architecture, only_consider_add=True):
        blocks = set()
        exclusions = []
        if only_consider_add:
            exclusions = ['elem_mul', 'sub']

        for key in architecture.blocks.keys():
            block = architecture.blocks[key]
            if len(block.inputs) == 2 and block.combination not in exclusions:
                blocks.add(block.identifier)

        return list(blocks)

    def get_output_key(self, architecture: Architecture, transformation):
        if transformation in self.transformation_ids.keys():
            prefix = self.transformation_ids[transformation]
        else:
            prefix = transformation

        arr = list(string.ascii_lowercase)
        output_key = str(list(architecture.get_all_block_keys())[0])
        count = 100
        while output_key in architecture.get_all_block_keys():
            output_key = f'{prefix}_' + arr[RandomGenerator.randint(0, len(arr) - 1)] + '_' + str(RandomGenerator.randint(1, count))
            count +=1
        return output_key

    def get_next_key(self, old_key, architecture: Architecture):
        count = 1
        new_key = old_key + '_' + str(count)
        while new_key in architecture.get_all_block_keys():
            count += 1
            new_key = old_key + '_' + str(count)
        return new_key

    def generate_add_single_unit(self, architecture: Architecture):
        blocks_to_select_from = copy.deepcopy(architecture.get_all_block_keys())
        blocks_to_select_from.remove('h_next')
        blocks_to_select_from.remove('c_next')
        blocks_to_select_from.remove('x')
        blocks_to_select_from.remove('h')
        blocks_to_select_from.remove('c')

        receiving_block_key = str(RandomGenerator.choice(blocks_to_select_from))
        receiving_block_inputs = []
        for inp in architecture.blocks[receiving_block_key].inputs:
            if type(inp) is not int and inp not in ['x', 'h', 'c']:
                receiving_block_inputs.append(inp)

        if len(receiving_block_inputs) == 0:
            receiving_block_key = str(RandomGenerator.choice(blocks_to_select_from))
            receiving_block_inputs = []
            for inp in architecture.blocks[receiving_block_key].inputs:
                if type(inp) is not int and inp not in ['x', 'h', 'c']:
                    receiving_block_inputs.append(inp)

            if len(receiving_block_inputs) == 0:
                return AddUnitTransformation(None, None, None, None)

        input_block_key = str(RandomGenerator.choice(receiving_block_inputs))

        options = EnvironmentConfig.get_config('activation_functions')
        distribution = ProbabilityDistribution.get_distribution(options)
        activation = list(distribution.keys())[0]
        output_key = self.get_output_key(architecture, 'add_unit')

        return AddUnitTransformation(input_block_key, output_key, activation, receiving_block_key)

    def generate_remove_singe_unit(self, architecture: Architecture) -> RemoveUnitTransformation:
        activation_layers = self.get_all_activation_layer_blocks(architecture)
        if len(activation_layers) == 0:
            LOG.info('State has 0 activation layers, skipping \'remove_single_unit\'.')
            return RemoveUnitTransformation(None)
        else:
            random_block_key = str(RandomGenerator.choice(activation_layers))
            return RemoveUnitTransformation(random_block_key)

    def generate_add_connection(self, architecture: Architecture, force_recurrent=False) -> AddConnectionTransformation:
        possible_input_blocks = copy.deepcopy(architecture.get_all_block_keys())
        possible_input_blocks.remove('h_next')
        possible_input_blocks.remove('c_next')

        if force_recurrent:
            possible_receiving_blocks = ['h_next', 'c_next']
        else:
            possible_receiving_blocks = copy.deepcopy(architecture.get_all_block_keys())
            possible_receiving_blocks.remove('x')
            possible_receiving_blocks.remove('h')
            possible_receiving_blocks.remove('c')

        input_block_key = str(RandomGenerator.choice(possible_input_blocks))
        input_block = architecture.blocks[input_block_key]

        receiving_block_key = str(RandomGenerator.choice(possible_receiving_blocks))
        receiving_block = architecture.blocks[receiving_block_key]

        count = 0
        while receiving_block_key in input_block.build_block_chain(
                architecture) or receiving_block_key in ['x', 'h', 'c'] or input_block_key in receiving_block.inputs:
            input_block_key = str(RandomGenerator.choice(possible_input_blocks))
            input_block = architecture.blocks[input_block_key]

            receiving_block_key = str(RandomGenerator.choice(possible_receiving_blocks))
            receiving_block = architecture.blocks[receiving_block_key]
            if count > 10:
                return AddConnectionTransformation(None, None, None, None)
            count += 1

        new_receiving_key = self.get_next_key(receiving_block_key, architecture)

        return AddConnectionTransformation(receiving_block_key, new_receiving_key, input_block_key, force_recurrent)

    def generate_remove_connection(self, architecture: Architecture) -> RemoveConnectionTransformation:
        blocks_to_select_from = copy.deepcopy(self.get_all_connected_blocks(architecture))
        static_blocks = ['h_next', 'c_next']
        for o_key in static_blocks:
            if o_key in blocks_to_select_from:
                blocks_to_select_from.remove(o_key)

        if len(blocks_to_select_from) == 0:
            LOG.info('State has 0 connected layers, skipping \'remove_connection\'.')
            return RemoveConnectionTransformation(None, None)

        def check_block_against_inputs(block_key):
            for block in architecture.blocks.keys():
                if block_key in architecture.blocks[block].inputs:
                    return True
            return False

        random_block_key = str(RandomGenerator.choice(blocks_to_select_from))
        while not check_block_against_inputs(random_block_key):
            blocks_to_select_from.remove(random_block_key)
            if len(blocks_to_select_from) == 0:
                LOG.info('Cannot find a block to remove a connection from.')
                return RemoveConnectionTransformation(None, None)
            random_block_key = str(RandomGenerator.choice(blocks_to_select_from))

        new_output = str(RandomGenerator.choice(architecture.blocks[random_block_key].inputs))
        return RemoveConnectionTransformation(random_block_key, new_output)

    def generate_change_random_activation(self, architecture: Architecture) -> ChangeRandomActivationTransformation:
        activation_layers = self.get_all_activation_layer_blocks(architecture, exclude_outputs=False)
        if len(activation_layers) == 0:
            LOG.info('State has 0 activation layers, skipping \'change_random_activation\'.')
            return ChangeRandomActivationTransformation(None, None, None)

        random_block_idx = RandomGenerator.randint(0, len(activation_layers) - 1)
        random_block_key = activation_layers[random_block_idx]

        original_activation = copy.deepcopy(architecture.blocks[random_block_key].activation[0])

        options = EnvironmentConfig.get_config('activation_functions')
        if original_activation in options:
            options.remove(original_activation)
        distribution = ProbabilityDistribution.get_distribution(options)
        activation = list(distribution.keys())[0]
        return ChangeRandomActivationTransformation(random_block_key, activation, original_activation)

    def generate_change_combination_connection(self,
                                               architecture: Architecture) -> ChangeCombinationConnectionTransformation:
        blocks_to_select_from = copy.deepcopy(self.get_all_connected_blocks(architecture, only_consider_add=False))

        if len(blocks_to_select_from) == 0:
            LOG.info('State has 0 connected layers, skipping \'change_combination_connection\'.')
            return ChangeCombinationConnectionTransformation(None, None, None)

        random_block_key = str(RandomGenerator.choice(blocks_to_select_from))
        if architecture.blocks[random_block_key].combination is None:
            block_combination = 'add'
        else:
            block_combination = architecture.blocks[random_block_key].combination

        options = EnvironmentConfig.get_config('combination_methods')
        if block_combination in options:
            options.remove(block_combination)

        distribution = ProbabilityDistribution.get_distribution(options)
        new_combination = list(distribution.keys())[0]
        return ChangeCombinationConnectionTransformation(random_block_key, new_combination, block_combination)

    def generate_add_gate_unit(self, architecture: Architecture):

        possible_receiving_blocks = []
        possible_receiving_blocks += architecture.blocks['h_next'].inputs
        possible_receiving_blocks += architecture.blocks['c_next'].inputs

        inputs_to_remove = ['x', 'h', 'c']

        for i in inputs_to_remove:
            while str(i) in possible_receiving_blocks:
                possible_receiving_blocks.remove(str(i))

        receiving_block_key = str(RandomGenerator.choice(possible_receiving_blocks))

        combination_options = EnvironmentConfig.get_config('combination_methods')
        combination = str(RandomGenerator.choice(combination_options))

        input_options = ['x', 'h', 'c']
        if combination == 'sub':
            # To allow for a gate similar to the GRU '1-z'
            input_options.append('1')

        input_1 = str(RandomGenerator.choice(input_options))
        input_options.remove(input_1)
        input_2 = str(RandomGenerator.choice(input_options))

        if input_2 == '1':
            input_2 = input_1
            input_1 = '1'

        transformation_inputs = [input_1, input_2]

        input_activations = [
            str(RandomGenerator.choice(['linear', 'linear_b'])),
            str(RandomGenerator.choice(['linear', 'linear_b']))
        ]

        activation_options = EnvironmentConfig.get_config('activation_functions')
        activation_options.remove('linear')
        activation_options.remove('linear_b')
        activation = str(RandomGenerator.choice(activation_options))

        inp_1_key = str(input_1) + '_' + self.get_output_key(architecture, 'new_gate_input')
        inp_2_key = str(input_2) + '_' + self.get_output_key(architecture, 'new_gate_input')
        combine_block_key = inp_1_key + '_' + combination + '_' + inp_2_key + '_' + self.get_random_strings()
        new_receiving_key = self.get_next_key(receiving_block_key, architecture)
        output_key = self.get_random_strings() + '_' + self.get_output_key(architecture, 'add_gate')
        return DifferentiableCombineActivateGateTransformation(transformation_inputs, input_activations,
                                                               combination, activation, output_key,
                                                               receiving_block_key,
                                                               inp_1_key, inp_2_key, combine_block_key,
                                                               new_receiving_key, 'add')

    def save_all_transformations(self, identifier):
        list_of_transformations = ListOfTransformations()
        for key in self.transformations.keys():
            transformation = self.transformations[key]
            self.persist_transformation_to_file(transformation)
            list_of_transformations.transformations.append(key)

        f = open(f'./output/transformations/list_of_transformations_{identifier}.json', 'w')
        json_object = jsonpickle.encode(list_of_transformations)
        f.write(json_object)
        f.close()

    def persist_transformation_list(self, transformation_list):
        if not EnvironmentConfig.get_post_slack():
            return
        for transformation in transformation_list:
            self.persist_transformation_to_file(transformation)

    def persist_transformation_to_file(self, transformation):
        file_name_path = f'./output/transformations/{transformation.hash}.json'
        if os.path.exists(file_name_path):
            print('Attempting to persist a transformation that has already been performed.')

        f = open(file_name_path, 'w')
        json_object = jsonpickle.encode(transformation)
        f.write(json_object)
        f.close()

    def load_transformation_from_file(self, identifier):
        f = open(f'./output/transformations/{identifier}.json')
        json_str = f.read()
        transformation = jsonpickle.decode(json_str)
        f.close()
        return transformation

    def read_transformations_from_file(self):
        f = open('./output/transformations/list_of_transformations.json')
        json_str = f.read()
        list_of_transformations = jsonpickle.decode(json_str)
        f.close()

        for key in list_of_transformations.transformations:
            self.transformations[key] = self.load_transformation_from_file(key)

        return list_of_transformations

    def get_random_strings(self):
        arr = list(string.ascii_lowercase)
        return f'{str(RandomGenerator.choice(arr))}{str(RandomGenerator.choice(arr))}_{str(RandomGenerator.choice(arr))}{str(RandomGenerator.choice(arr))}'

    def get_random_activation(self) -> str:
        activation_options = EnvironmentConfig.get_config('activation_functions')
        activation_distribution = ProbabilityDistribution.get_distribution(activation_options)
        return list(activation_distribution.keys())[0]

    def get_random_combination(self) -> str:
        combination_options = EnvironmentConfig.get_config('combination_methods')
        combination_distribution = ProbabilityDistribution.get_distribution(combination_options)
        return list(combination_distribution.keys())[0]

    def post_linear_counts(self):
        message = f'Done {self.linear_counts} linear counts and {self.non_linear_counts} non-linear counts. Added {self.gates_added} gates. Added {self.x_linear_connection} x-linear connections.'
        LOG.info(message)
        # SlackPost.post_neutral('Linear counts', message)
        self.linear_counts = 0
        self.non_linear_counts = 0
        self.gates_added = 0
        self.x_linear_connection = 0
