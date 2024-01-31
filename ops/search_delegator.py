import copy
import glob
import json
import os
import time
import zipfile

import jsonpickle
import torch

from config.env_config import EnvironmentConfig
from example_datasets.cfl.language_generator import Language_Generator
from example_datasets.cfl.language_model_objectives import LanguageModelObjectives
from example_datasets.cfl.language_model_trainer import Language_Model_Trainer
from example_datasets.ptb.ptb_model_objectives import PtbModelObjectives
from example_datasets.ptb.ptb_trainer import PtbTrainer
from example_datasets.sentiment.sentiment_data_loader import SentimentDataLoader
from example_datasets.sentiment.sentiment_model_wrapper import SentimentModel
from example_datasets.sentiment.sentiment_trainer import SentimentTrainer
from model.architecture import Architecture
from model.architecture_objectives import ArchitectureObjectives
from model.circular_reference_exception import CircularReferenceException
from model.exploding_gradient_exception import ExplodingGradientException
from model.model_fitness import ModelFitness
from ops.block_state_builder import BlockStateBuilder
from ops.block_transformation import BlockTransformation
from ops.compare_architecture_performances import CompareArchitecturePerformances
from ops.nas_controller import NASController
from ops.nondominated_sorting import NonDominatedSorting
from ops.objective_evaluation import ObjectiveEvaluation
from ops.probability_distribution import ProbabilityDistribution
from persistence.model_persistence import ModelPersistence
from persistence.persistence import Persistence
from utils.logger import LOG
from utils.random_generator import RandomGenerator
from utils.slack_post import SlackPost
from utils.tensorboard_writer import TensorBoardWriter

LOG = LOG.get_instance().get_logger()

GOOGLE_DRIVE_EXISTS = os.path.exists('/content/drive/My Drive/msc_run')
RESTORE = f'./restore' if not GOOGLE_DRIVE_EXISTS else f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}/restore'
OUTPUT = f'./output' if not GOOGLE_DRIVE_EXISTS else f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}/output'

class SearchDelegator:
    """
    This is the main class which delegates the searching for RNN architectures.
    """

    def __init__(self):
        self.generation = 0

        self.architectures = {}
        self.arch_history = {}

        self.fronts = {}
        self.crowding_distances = {}

        self.population_fitness = {}
        self.fitness_history = {}
        self.architecture_transformation_history = {}

        self.search_start_time = time.time()
        self.generation_times = []

        self.population = set()
        self.selected = []
        self.children = []

        # Chu, X., & Yu, X. (2018). Improved Crowding Distance for NSGA-II. http://arxiv.org/abs/1811.12667
        self.use_alternate_nsga_crowding = self.config('alternative_nsga_crowding')

    def search(self):
        population_size = self.config('population_size')
        population_to_select = self.config('population_to_select')
        SlackPost.post_neutral('Searching started',
                               f'For population size={population_size}; and {self.config("number_of_generations")} generations.')

        self.clean_up_arch_files()
        self.restore_snapshot()
        self.clean_up_architectures()

        transformer = BlockTransformation(generation=self.generation)
        self.initialise_population(population_size, transformer)
        transformer.post_linear_counts()

        self.save_architectures()

        try:
            self.evaluate_objective_functions(list(self.population))
        except TimeExceededException:
            SlackPost.post_neutral('Time Exceeded', 'Time exceeded, returning.')
            self.finalise_search()
            return

        if self.config('number_of_generations') > 0:
            end_generation = self.generation + self.config('number_of_generations')

            self.non_dominated_sorting(list(self.population))
            self.print_current_best_architectures()

            architectures_to_select_from = self.get_architecture_to_select_from()
            LOG.info(f'architectures_to_select_from = {architectures_to_select_from}')

            if len(self.selected) == 0 or len(self.children) == 0:

                if len(architectures_to_select_from) == population_size:
                    self.selected = architectures_to_select_from
                else:
                    self.selected = self.perform_selection_by_rank(population_to_select, architectures_to_select_from)

                if transformer.generation == 0:
                    transformer.generation = 1
                self.children = self.generate_children(self.selected, transformer)
                self.clean_up_architectures()

            for i in range(self.config('number_of_generations')):
                if self.check_times():
                    break

                gen_start = time.time()
                LOG.info('=' * 72)
                LOG.info(f'Generation {self.generation + 1}')
                LOG.info(f'Fronts:: {self.fronts}')
                LOG.info('=' * 72)

                # SlackPost.post_neutral(f'Generation {self.generation + 1} / {end_generation}',
                #                        f'Parents: {self.selected}\nChildren: {self.children}.')
                SlackPost.post_neutral(f'Generation {self.generation + 1}', f'Population size :: {len(self.population)}')
                self.save_architectures()

                try:
                    self.evaluate_objective_functions(list(self.population))
                except TimeExceededException:
                    SlackPost.post_neutral('Time Exceeded', 'Time exceeded, returning.')
                    self.finalise_search()
                    return

                self.non_dominated_sorting(list(self.population))
                self.update_population(population_size)
                self.print_current_best_architectures()
                self.log_current_metrics()

                self.generation += 1
                if i + 1 < self.config('number_of_generations') and not os.path.exists(f'{OUTPUT}/kill.txt'):

                    architectures_to_select_from = self.get_architecture_to_select_from()
                    if len(architectures_to_select_from) == population_size:
                        self.selected = architectures_to_select_from
                    else:
                        self.selected = self.perform_selection_by_rank(population_to_select,
                                                                       architectures_to_select_from)
                        for s in self.selected:
                            if s in self.population_fitness.keys():
                                LOG.info(f'[{s}] = {self.population_fitness[s]}')

                    transformer.generation = self.generation
                    self.children = self.generate_children(self.selected, transformer)
                    self.clean_up_architectures()
                    transformer.post_linear_counts()
                else:
                    self.selected = []
                    self.children = []

                diff = time.time() - gen_start
                self.generation_times.append(diff)

                if os.path.exists(f'{OUTPUT}/kill.txt'):
                    LOG.info(f'Trigger found, exiting.')
                    SlackPost.post_neutral('Exit trigger', 'Trigger found, exiting.')
                    break

            LOG.info(
                f'Final population :: {list(self.architectures.keys())} = children: {self.children} | parents: {self.selected}.')
        self.finalise_search()

    def update_population(self, population_size):
        architectures = self.get_architecture_to_select_from()
        self.population = set(architectures[:population_size])

    def initialise_population(self, population_size: int, transformer: BlockTransformation):
        """
        Initialisation of the initial population.

        Use the env.json configuration file to specify the seed architectures, which is limited to the basic RNN (with tanh activation),
        LSTM and GRU architectures.

        :param population_size: the total number of architectures to generate.
        :param transformer: the transformer instance that will be used for generating random architectures.
        :return: N/A
        """
        builder = BlockStateBuilder('BASIC')

        if len(self.architectures.keys()) > 0 and len(self.architectures.keys()) < self.config('population_size'):
            LOG.info(f'{len(self.architectures.keys())} architectures loaded, '
                     f'{self.config("population_size") - len(self.architectures.keys())} new architectures to be generated.')

        if 'BASIC_0' not in self.architectures.keys() and self.should_model_be_loaded('BASIC_0'):
            self.architectures['BASIC_0'] = builder.get_basic_architecture()
            self.architectures['BASIC_0'].identifier = 'BASIC_0'
            self.population.add('BASIC_0')

        if self.config('include_gru'):
            if self.should_model_be_loaded('GRU_0'):
                self.architectures['GRU_0'] = builder.get_gru_architecture()
                self.architectures['GRU_0'].identifier = 'GRU_0'
                self.population.add('GRU_0')
            else:
                LOG.info(
                    f'GRU architecture loaded. {"Fitness for GRU loaded." if "GRU_0" in self.population_fitness.keys() else "Could not load fitness for GRU."}')
        elif 'GRU_0' in self.architectures.keys():
            LOG.info(f'GRU was restored but is excluded from loaded config, removing GRU from architectures.')
            self.architectures.pop('GRU_0')
            if 'GRU_0' in self.population:
                self.population.remove('GRU_0')

        if self.config('include_lstm'):
            if self.should_model_be_loaded('LSTM_0'):
                self.architectures['LSTM_0'] = builder.get_lstm_architecture()
                self.architectures['LSTM_0'].identifier = 'LSTM_0'
                self.population.add('LSTM_0')
            else:
                LOG.info(
                    f'LSTM architecture loaded. {"Fitness for LSTM loaded." if "LSTM_0" in self.population_fitness.keys() else "Could not load fitness for LSTM."}')
        elif 'LSTM_0' in self.architectures.keys():
            LOG.info(f'LSTM was restored but is excluded from loaded config, removing LSTM from architectures.')
            self.architectures.pop('LSTM_0')
            if 'LSTM_0' in self.population:
                self.population.remove('LSTM_0')

        LOG.info(f'Population size = {population_size}, current population size = {len(self.population)}.')
        if len(self.population) == population_size:
            return

        self.sync_arch_history()

        population_size -= len(self.population)
        if population_size > 0:
            LOG.info(f'Generating {population_size} new architectures.')
            for i in range(population_size):
                highest = self.get_highest_arch_key()
                self.generate_random_architecture(highest, transformer)

    def generate_random_architecture(self, count, transformer: BlockTransformation):
        new_cell = None
        transformation_count = None
        while new_cell is None or self.test_new_architecture_similarity(new_cell):
            activation_function = ProbabilityDistribution.get_activation_function()
            new_cell = BlockStateBuilder.get_basic_architecture(activation=activation_function)
            transformer.transform_architecture(new_cell, transformation_count=transformation_count)
            if transformation_count is None:
                transformation_count = 12
            else:
                transformation_count += 1

        new_cell.identifier = f'rdm{"j" if not new_cell.elman_network else ""}{count}_0'
        new_count = copy.copy(count)
        while new_cell.identifier in self.architectures.keys():
            new_count += 1
            new_cell.identifier = f'rdm{"j" if not new_cell.elman_network else ""}{new_count}_0'

        self.architectures[new_cell.identifier] = new_cell
        self.population.add(new_cell.identifier)
        return new_cell.identifier

    def finalise_search(self):
        """
        This method finalises the search by persisting the architectures and saving a snapshot of the current search state.
        :return:
        """
        self.save_architectures()
        self.save_snapshot()
        TensorBoardWriter.close_writer('search')
        diff = (time.time() - self.search_start_time) / 60
        LOG.info(f'Search took {diff} minutes.')

    def non_dominated_sorting(self, architecture_keys: list):
        with NonDominatedSorting(self.population_fitness, self.config('objectives'), EnvironmentConfig.get_config('dataset')=='sentiment') as sorter:
            self.fronts = sorter.sort(architecture_keys)

    def perform_selection_by_rank(self, population_size, options) -> list:

        candidate_fronts = {}
        keys_to_select_from = []
        for key in options:
            keys_to_select_from.append(key)
            for i in self.fronts.keys():
                if key in self.fronts[i]:
                    candidate_fronts[key] = i

        if len(keys_to_select_from) < population_size:
            return keys_to_select_from

        selection = []
        while len(selection) < population_size:
            if len(keys_to_select_from) == 1:
                selection.append(keys_to_select_from[0])
                break

            parent_1 = RandomGenerator.choice(keys_to_select_from)
            parent_2 = RandomGenerator.choice(keys_to_select_from)
            while parent_1 == parent_2:
                parent_2 = RandomGenerator.choice(keys_to_select_from)

            # Attempt selection based on domination (lower is better).
            if candidate_fronts[parent_1] < candidate_fronts[parent_2]:
                keys_to_select_from.remove(parent_1)
                selection.append(parent_1)
            elif candidate_fronts[parent_2] < candidate_fronts[parent_1]:
                keys_to_select_from.remove(parent_2)
                selection.append(parent_2)

            # Alternatively, perform selection based on crowding distance
            else:
                c_dist_key = candidate_fronts[parent_1]
                if c_dist_key not in self.crowding_distances.keys():
                    self.crowding_distance(c_dist_key)
                crowding_dist = self.crowding_distances[c_dist_key]

                if parent_1 in crowding_dist.keys() and parent_2 in crowding_dist.keys():
                    if crowding_dist[parent_1] > crowding_dist[parent_2]:
                        keys_to_select_from.remove(parent_1)
                        selection.append(parent_1)
                    else:
                        keys_to_select_from.remove(parent_2)
                        selection.append(parent_2)
                elif parent_1 in crowding_dist.keys():
                    keys_to_select_from.remove(parent_1)
                    selection.append(parent_1)
                else:
                    keys_to_select_from.remove(parent_2)
                    selection.append(parent_2)

        keys_to_remove = []
        for key in self.architectures.keys():
            if key not in options:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self.remove_architecture(key)

        return selection

    def crowding_distance(self, front_key: int):
        """
        Calculate the crowding distance assignment for the provided front.
        :param front_key:
        :return:
        """
        distances = {}
        for i in self.fronts[front_key]:
            distances[i] = 0

        for m in self.config('objectives'):
            sorted_obj_distances, idx_1, idx_N = self.get_sorted_distances_for_objective(front_key, m)
            max_m = list(sorted_obj_distances[0])[0]
            min_m = list(sorted_obj_distances[-1])[0]
            distances[idx_1] = float('inf')
            distances[idx_N] = float('inf')

            max_min_delta = (max_m - min_m)
            if max_min_delta == 0:
                max_min_delta = 1

            for i, v in enumerate(sorted_obj_distances):
                (value, key) = v
                if key not in distances.keys():
                    distances[key] = value
                if key != idx_1 and key != idx_N:
                    if self.use_alternate_nsga_crowding:
                        distances[key] = distances[key] + (
                                (sorted_obj_distances[i + 1][0] - sorted_obj_distances[i][0]) / max_min_delta)
                    else:
                        distances[key] = distances[key] + (
                                (sorted_obj_distances[i + 1][0] - sorted_obj_distances[i - 1][0]) / max_min_delta)

        self.crowding_distances[front_key] = distances

    def get_sorted_distances_for_objective(self, front_key: int, objective: str) -> (list, str, str):
        """
        Compiles a list of the population which is sorted based on the provided objective.
        Minimisation is assumed; thus the list returned is sorted in ascending order.

        The keys of the lowest (idx_1) and highest (idx_N) objective-valued individuals are also returned.
        :param front_key:
        :param objective:
        :return:
        """

        if objective not in self.config('objectives'):
            raise Exception(f'Unknown objective {objective} specified.')

        values = []
        front = copy.deepcopy(self.fronts[front_key])
        for key in front:
            value = getattr(self.population_fitness[key], objective)
            values.append((value, key))

        is_sentiment = objective == 'ptb_ppl' and EnvironmentConfig.get_config('dataset') == 'sentiment'
        values = sorted(values, key=lambda x: (x[0]), reverse=is_sentiment)
        idx_1 = list(values[0])[1]
        idx_N = list(values[-1])[1]

        return values, idx_1, idx_N

    def merge_population(self, parents: list, front_key: int) -> list:
        result = parents
        result += list(copy.copy(self.fronts[front_key]))
        return result

    def get_architecture_to_select_from(self) -> list:
        """
        This method returns a list of architectures to select from.

        More complicated list compilation can be implemented here if required, as was done in:
        Elsken, T., Metzen, J. H., & Hutter, F. (2018). Efficient multi-objective neural architecture search
        via lamarckian evolution. ArXiv, 1–23.
        :return:
        """
        keys = []
        for _f in self.fronts.keys():
            for _k in self.fronts[_f]:
                keys.append(_k)
        return keys

    def evaluate_objective_functions(self, population: list):
        """
        This method is responsible for evaluating the fitness of the current population.

        :param population:
        :return:
        """

        LOG.info(f'{">" * 50} EVALUATE :: {population} {"<" * 50}')
        training_times = []
        for key in population:
            if key in self.population_fitness.keys() and not self.check_if_fitness_default(key):
                LOG.info(f'Fitness for {key} already exist.')
                continue
            elif key in self.fitness_history.keys():
                LOG.info(f'{key} fitness in history, can continue.')
                continue

            if len(training_times) > 0:
                avg_time = sum(training_times) / len(training_times)
                diff = time.time() - self.search_start_time

                # Allows for a run time limit to be specified, this will terminate the search when the time limit has been reached.
                if -1 < self.config('time_limit') < (diff + avg_time):
                    LOG.info(
                        f'Average training time of {avg_time} and current running time {diff} exceeds specified time limit.')
                    raise TimeExceededException

            self.evaluating_architecture = key
            previous_snapshot = self.save_snapshot()
            architecture = self.architectures[key]
            if key not in self.population_fitness.keys():
                self.population_fitness[key] = ModelFitness(key)

            _, number_of_blocks, number_of_mul, _ = \
                ObjectiveEvaluation.evaluate_cheap_objectives(architecture, key)
            self.population_fitness[key].number_of_blocks = number_of_blocks
            self.population_fitness[key].number_of_mul = number_of_mul

            start_time = time.time()
            self.get_model_expensive_objectives(key)
            evaluation = time.time() - start_time
            training_times.append(evaluation)

            self.evaluating_architecture = None
            self.save_snapshot(add_count=1)
            if os.path.exists(previous_snapshot):
                os.remove(previous_snapshot)

    def get_model_expensive_objectives(self, cell_key):
        """
        This is where the models are trained and tested on the provided dataset (expensive to evaluate).

        See PtbTrainer in the example_datasets.ptb.PtbTrainer for an example implementation.

        :param cell_key:
        :return:
        """

        TensorBoardWriter.create_new_writer(cell_key)

        if EnvironmentConfig.get_config('dataset') == 'ptb':
            test_loss, perplexity, performance = self.get_ptb_perplexity(cell_key)
        elif EnvironmentConfig.get_config('dataset') == 'sentiment':
            test_loss, perplexity, performance = self.get_sentiment_fitness(cell_key)
        else:
            test_loss, perplexity, performance = self.get_cfl_fitness(cell_key)

        self.population_fitness[cell_key].ptb_loss = test_loss
        self.population_fitness[cell_key].ptb_ppl = perplexity
        self.population_fitness[cell_key].ptb_performance = performance

        if len(performance.get('training_time', [])) > 0:
            self.population_fitness[cell_key].training_time = sum(performance['training_time']) / len(
                performance['training_time'])

        is_new_best = Persistence.is_new_best(self.population_fitness[cell_key], EnvironmentConfig.get_config('dataset'))
        if len(is_new_best) != 0:
            LOG.info(f'{cell_key} has achieved a new best for one of the objectives {is_new_best}.')
            SlackPost.post_success('New best',
                                   f'{cell_key} has achieved a new best for one of the objectives {is_new_best}.')

        TensorBoardWriter.close_writer(cell_key)
        _hash = json.dumps(self.architectures[cell_key].get_block_strings())
        Persistence.persist_model_performance_objectives(cell_key, str(hash(_hash)))

    def get_ptb_perplexity(self, cell_key):
        """
        This method trains and tests a model created for the provided architecture on the Penn Treebank dataset to
        obtain the test perplexity achieved. The test perplexity is assumed to be the architecture performance
        objective in this example implementation.

        :param cell_key:
        :return:
        """

        parent_performance = {}
        warm_start_parent = None
        ptb_epochs = self.config('ptb_training_epochs')

        parent_key = self.architectures[cell_key].get_parent()
        if parent_key is not None:
            warm_start_parent = parent_key if self.config('warm_start_models') else None
            if parent_key in self.architectures.keys():
                parent_architecture = self.architectures[parent_key]
                if parent_key in self.population_fitness.keys():
                    parent_performance = self.population_fitness[parent_key].ptb_performance
            else:
                parent_architecture = self.restore_architecture(parent_key)

            # Calculate the perplexity difference between parent and offspring architecture
            loss_difference, ppl_difference = CompareArchitecturePerformances.compare_parent_child_ptb(
                parent_architecture, self.architectures[cell_key])

            # If the perplexity difference between the parent and offspring architecture is within the specified
            # threshold, we can reduce the number of epochs to train the offspring architecture for.
            LOG.debug(f'Diff for parent ({parent_key}) and child ({cell_key}) = {ppl_difference}')
            if ppl_difference < self.config('ptb_compare_ppl_threshold'):
                ptb_epochs = self.config('ptb_threshold_epochs')
                LOG.info(f'Child epochs adjusted to {ptb_epochs}.')
        else:
            LOG.debug(f'Could not find parent for {cell_key}')

        trainer_arch = copy.deepcopy(self.architectures[cell_key])
        best_ppl = Persistence.get_current_best('ptb_ppl')
        with PtbTrainer(trainer_arch, cell_key, ptb_epochs, nlayers=self.config('ptb_model_nlayers'),
                        best_ppl=best_ppl) as ptb_trainer:

            model = None
            if os.path.exists(f'{OUTPUT}/models/{cell_key}.tar'):
                model = ptb_trainer.build_model()
                ModelPersistence.load_model(cell_key, model)
                LOG.debug(f'Loaded existing model for {cell_key}.')

            test_loss, perplexity, performance = ptb_trainer.run(parent_performance, model=model,
                                                                 force_training=self.config('force_ptb_training'),
                                                                 warm_start_parent=warm_start_parent,
                                                                 override_train_data=self.config('override_train_data'))

            self.population_fitness[cell_key].number_of_parameters = ptb_trainer.model_params

            NASController.set_architecture_performance(cell_key, PtbModelObjectives.PTB_MODEL_LOSS, test_loss)
            NASController.set_architecture_performance(cell_key, PtbModelObjectives.PTB_MODEL_PPL, perplexity)
            NASController.set_architecture_performance(cell_key, ArchitectureObjectives.NUMBER_OF_BLOCKS, len(self.architectures[cell_key].blocks.keys()))
            NASController.set_architecture_performance(cell_key,
                                                       ArchitectureObjectives.NUMBER_OF_PARAMETERS,
                                                       ptb_trainer.model_params)

        return test_loss, perplexity, performance

    def get_sine_wave_loss(self, cell_key):

        LOG.info(f'Training model with SINE wave dataset, {cell_key}')

        builder = BlockStateBuilder(cell_key)
        model = builder.build_sine_model(self.architectures[cell_key], 1, 51)
        self.population_fitness[cell_key].number_of_parameters = model.get_parameters()

        parent_performance = {}
        parent_key = self.architectures[cell_key].get_parent()
        sine_epochs = None
        if parent_key is not None:
            child_architecture = self.architectures[cell_key]
            if parent_key not in self.architectures.keys():
                parent_architecture = self.restore_architecture(parent_key)
            else:
                parent_architecture = self.architectures[parent_key]

            """
            loss_difference = CompareArchitecturePerformances.compare_parent_child_sine(parent_architecture, child_architecture)
            LOG.info(f'Difference :: {loss_difference}')
            if loss_difference < self.config['sine_loss_threshold']:
                sine_epochs = 5
                LOG.info(f'Epochs adjusted for {cell_key}')
            """

            if parent_key in self.population_fitness.keys():
                parent_performance = self.population_fitness[parent_key].sine_wave_performance

        if self.config('warm_start_models') and parent_key is not None:
            if os.path.exists(f'{OUTPUT}/models/{parent_key}.tar'):
                ModelPersistence.load_model(f'{parent_key}', model)
                LOG.info(f'Warm started {cell_key} from parent {parent_key}.')

        try:
            with SineWaveTrainer(epochs=sine_epochs) as trainer:
                train_start_time = time.time()
                loss, mean_absolute_error, performance = trainer.train_model(model, cell_key,
                                                                             cell_key,
                                                                             self.generation,
                                                                             parent_performance,
                                                                             number_of_phases=self.config(
                                                                                 'sine_model_phases'),
                                                                             noise_range=self.config(
                                                                                 'sine_model_noise_range'))
                training_time = time.time() - train_start_time
        except ExplodingGradientException:
            LOG.info(f'Gradient exploded for {cell_key}.')
            return 1.0e+10, 1.0e+10, {}

        return loss, mean_absolute_error, performance

    def get_sentiment_fitness(self, cell_key):

        LOG.info('Start sentiment')
        data_loader = SentimentDataLoader()
        no_layers = 1
        vocab_size = len(data_loader.vocab) + 1  # extra 1 for padding
        embedding_dim = 64
        output_dim = 1
        hidden_dim = embedding_dim

        architecture = self.architectures[cell_key]
        builder = BlockStateBuilder(cell_key)
        model = SentimentModel(architecture, cell_key, builder, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, lstm_model=False, basic_rnn=False)

        parent_key = self.architectures[cell_key].get_parent()
        if parent_key is not None and self.config('warm_start_models'):
            ModelPersistence.load_model(f'{parent_key}_sentiment', model)

        trainer = SentimentTrainer()
        LOG.info(f'Start sentiment training {cell_key}')
        accuracy, performance = trainer.train(model, data_loader)

        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            total_params += param

        NASController.set_architecture_performance(cell_key, PtbModelObjectives.PTB_MODEL_LOSS, accuracy)
        NASController.set_architecture_performance(cell_key, PtbModelObjectives.PTB_MODEL_PPL, accuracy)
        NASController.set_architecture_performance(cell_key, ArchitectureObjectives.NUMBER_OF_BLOCKS, len(self.architectures[cell_key].blocks.keys()))
        NASController.set_architecture_performance(cell_key, ArchitectureObjectives.NUMBER_OF_PARAMETERS, total_params)

        LOG.info(f'Done sentiment training {cell_key}')
        return accuracy, accuracy, performance


    def get_cfl_fitness(self, cell_key):
        self.language = 'abc'
        LOG.info(f'Training model with {self.language} regular language dataset, {cell_key}')
        language_generator = Language_Generator(self.language)

        builder = BlockStateBuilder(cell_key)
        inp_size = len(self.language) + 1
        hid_size = 128
        model = builder.build_cfl_model(self.architectures[cell_key], inp_size,
                                                        hid_size, output_layer_dimension=inp_size)
        self.population_fitness[cell_key].number_of_parameters = model.get_parameters()
        model.identifier = cell_key
        parent_performance = {}
        parent_key = self.architectures[cell_key].get_parent()
        lang_model_epochs = self.config('language_model_epochs')
        if parent_key is not None:
            warm_start_parent = parent_key if self.config('warm_start_models') else None

            if parent_key in self.population_fitness.keys():
                parent_performance = self.population_fitness[parent_key].lang_performance

            if warm_start_parent is not None:
                if os.path.exists(f'{OUTPUT}/models/{warm_start_parent}.tar'):
                    ModelPersistence.load_model(f'{warm_start_parent}', model)
                    LOG.info(f'Warm started {cell_key} from parent {warm_start_parent}.')
                elif os.path.exists(f'{OUTPUT}/models/{warm_start_parent}_lang.tar'):
                    ModelPersistence.load_model(f'{warm_start_parent}_lang', model)
                    LOG.info(f'Warm started {cell_key} from parent {warm_start_parent}.')
                else:
                    LOG.info(
                        f'Warm start enabled for {cell_key} but parent {warm_start_parent} could not be found.')

        try:
            train_start_time = time.time()
            with Language_Model_Trainer(epochs=lang_model_epochs) as trainer:
                loss, performance = trainer.train_model(model, language_generator, cell_key,
                                                        parent_performance)
                training_time = time.time() - train_start_time
                NASController.set_architecture_performance(cell_key,
                                                           LanguageModelObjectives.LANG_MODEL_LOSS,
                                                           loss)
                NASController.set_architecture_performance(cell_key,
                                                           ArchitectureObjectives.NUMBER_OF_PARAMETERS,
                                                           model.get_parameters())
                NASController.set_architecture_performance(cell_key,
                                                           ArchitectureObjectives.TRAINING_TIME,
                                                           training_time)

                return loss, loss, performance
        except ExplodingGradientException:
            LOG.info(f'Gradient exploded for {cell_key}.')
            model.clear_outputs()
            return float("inf"), float("inf"), {}


    def generate_children(self, parents, transformer) -> list:
        """
        This method generates offspring for the provided parent architectures.

        For each parent architecture, a single offspring architecture is created.

        The parent architecture is cloned to create the offspring architecture. Network transformations are then performed
        on the offspring architecture.

        :param parents:
        :param transformer:
        :return:
        """

        parents_to_select_from = copy.copy(parents)

        children = []
        parents_buffer = []
        for p in parents_to_select_from:

            child_arch = None
            counter = 0
            exceptions = []
            while child_arch is None and counter < 10:
                try:
                    child_arch = copy.deepcopy(self.architectures[p])
                    child_key = self.get_next_key(p)
                    child_arch.identifier = child_key
                    child_arch.ancestors.append(p)

                    if self.architectures[p].main_parent is None:
                        child_arch.main_parent = p

                    transformer.transform_architecture(child_arch)
                    self.architectures[child_key] = child_arch
                    children.append(child_key)
                    parents_buffer.append(p)
                except CircularReferenceException as e:
                    LOG.info(f'Generate child for {p} :: {e}')
                    exceptions.append(e)
                    child_arch = None
                    counter += 1

            if child_arch is None:
                SlackPost.post_failure('Offspring generation', f'Unable to generate offspring for {p}; too many recursive encounters.')
                LOG.info(exceptions)

                counter = 0
                exceptions = []
                while child_arch is None and counter < 10:
                    parent = RandomGenerator.choice(parents_buffer)
                    try:
                        child_arch = copy.deepcopy(self.architectures[parent])
                        child_key = self.get_next_key(parent)
                        child_arch.identifier = child_key
                        child_arch.ancestors.append(parent)

                        if self.architectures[parent].main_parent is None:
                            child_arch.main_parent = parent

                        transformer.transform_architecture(child_arch)
                        self.architectures[child_key] = child_arch
                        children.append(child_key)
                    except CircularReferenceException as e:
                        LOG.info(f'Generate child for {parent} :: {e}')
                        exceptions.append(e)
                        child_arch = None
                        counter += 1

                if len(exceptions) > 5:
                    activation_function = ProbabilityDistribution.get_activation_function()
                    new_cell = BlockStateBuilder.get_basic_architecture(activation=activation_function)
                    child_key = self.get_next_key(p)
                    new_cell.identifier = child_key
                    transformer.transform_architecture(new_cell, transformation_count=self.config('initial_population_transformations'), initial_generation=True)
                    self.architectures[child_key] = child_arch
                    children.append(child_key)
                    SlackPost.post_neutral('Offspring generation', f'Generated a new architecture for {child_key}.')

            if child_arch is not None:
                self.population.add(child_arch.identifier)

        return children

    def restore_snapshot(self):
        """
        During the search process, snapshots are saved so that the search can be continued at a later stage.
        This method restores the state of the search to the most recent snapshot if it exist.
        :return:
        """
        if not os.path.exists(RESTORE) or not self.config('restore_if_possible'):
            LOG.info('Not restoring.')
            return False

        files = glob.glob(f'{RESTORE}/snapshot_*_*.pt', recursive=False)

        if len(files) == 0:
            LOG.info('Attempted restore, but no files were found to restore from.')
            return False

        versions = self.get_restore_versions(files)

        snapshot = torch.load(f'{RESTORE}/snapshot_{versions[0][0]}_{versions[0][1]}.pt')
        LOG.info(f'Restoring snapshot version {versions[0][0]}_{versions[0][1]}.')
        self.generation = snapshot['generation']
        self.fronts = snapshot['fronts']
        self.population_fitness = snapshot['population_fitness']
        self.architecture_transformation_history = snapshot['architecture_transformation_history']
        self.fitness_history = snapshot['fitness_history']
        self.selected = snapshot['selected']
        self.children = snapshot['children']
        self.arch_history = snapshot['arch_history']
        self.population = snapshot['population']
        try:
            self.evaluating_architecture = snapshot['evaluating_architecture']
        except KeyError as e:
            self.evaluating_architecture = None
        for key in snapshot['architecture_keys']:
            self.architectures[key] = self.restore_architecture(key)

        architectures_to_evaluate = []
        for key in self.architectures.keys():
            if key not in self.population_fitness.keys():
                if key in self.fitness_history.keys():
                    (ptb_loss, ptb_ppl, ptb_time) = self.fitness_history[key]
                    new_fitness = ModelFitness(key)
                    new_fitness.ptb_loss = ptb_loss
                    new_fitness.ptb_ppl = ptb_ppl
                    new_fitness.ptb_time = ptb_time
                    self.population_fitness[key] = new_fitness
                else:
                    LOG.info(f'WARNING :: architecture loaded for [{key}], but no associated fitness found.')
                    architectures_to_evaluate.append(key)

        if len(architectures_to_evaluate) > 0:
            LOG.info(f'Evaluating architectures without fitness : {architectures_to_evaluate}')
            self.evaluate_objective_functions(architectures_to_evaluate)

        self.sync_model_performances()
        restore_message = f'Restored {len(self.architectures.keys())} / {len(self.population_fitness.keys())} architectures from snapshot_{versions[0][0]}_{versions[0][1]}.'
        LOG.info(restore_message)
        # SlackPost.post_neutral('Restore', restore_message)
        return True

    def save_snapshot(self, add_count=0):
        """
        This method saves a snapshot of the current search state so that it can be restored later if required.
        :return:
        """

        if not os.path.exists(RESTORE):
            os.makedirs(RESTORE)

        count = len(glob.glob(f'{RESTORE}/snapshot_{self.generation}_*.pt', recursive=True)) + add_count
        snapshot_path = f'{RESTORE}/snapshot_{self.generation}_{count}.pt'
        torch.save({
            'generation': self.generation,
            'fronts': self.fronts,
            'population_fitness': self.population_fitness,
            'architecture_keys': list(self.architectures.keys()),
            'architecture_transformation_history': self.architecture_transformation_history,
            'fitness_history': self.fitness_history,
            'evaluating_architecture': self.evaluating_architecture,
            'selected': self.selected,
            'children': self.children,
            'arch_history': self.arch_history,
            'population': self.population
        }, snapshot_path)
        LOG.debug('Successfully updated snapshot.')

        for _arch in self.architectures.keys():
            if not os.path.exists(f'{restore_path}/architectures/{_arch}.json'):
                f = open(f'{restore_path}/architectures/{_arch}.json', 'w')
                json_object = jsonpickle.encode(self.architectures[_arch])
                f.write(json_object)
                f.close()
                LOG.debug(f'Saved architecture {_arch}')

        # self.zip_files()

        return snapshot_path

    def zip_files(self):
        if os.path.exists('/content/drive/My Drive/msc_run'):
            folders = []
            if os.path.exists('./config/'):
                folders.append('./config/')

            if os.path.exists('./output/'):
                folders.append('./output/')

            if os.path.exists('./restore/'):
                folders.append('./restore')

            if os.path.exists('./performance'):
                folders.append('./performance')

            if len(folders) > 0:
                self.zipit(folders, f'/content/drive/My Drive/msc_run/{EnvironmentConfig.get_config("dataset")}.zip')

    def zipit(self, folders, zip_filename):
        zip_file = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

        count = 0
        for folder in folders:
            for dirpath, dirnames, filenames in os.walk(folder):
                for filename in filenames:
                    zip_file.write(
                        os.path.join(dirpath, filename),
                        os.path.relpath(os.path.join(dirpath, filename), os.path.join(folders[0], '../..')))
                    count += 1

        zip_file.close()
        LOG.debug(f'Saved {count} files to Google Drive.')

    def get_restore_versions(self, files=None, generation_specific=None):
        """
        This method looks at all the available snapshots that are available and returns the most recent one.

        :param generation_specific:
        :param files:
        :return:
        """

        if files is None:
            files = glob.glob(f'{RESTORE}/snapshot_{generation_specific if generation_specific else "*"}_*.pt', recursive=False)

            if len(files) == 0:
                return []

        versions = []
        for file in files:
            first_pos = file.find('_')
            second_pos = file.find('_', first_pos + 1)
            second_end = file.find('.', second_pos)

            first_num = file[first_pos + 1:second_pos]
            second_num = file[second_pos + 1:second_end]

            versions.append((int(first_num), int(second_num)))

        versions = sorted(versions, key=lambda x: (x[0], x[1]), reverse=True)
        return versions

    def save_architectures(self):
        average_blocks = []
        for _arch in self.architectures.keys():
            average_blocks.append(len(self.architectures[_arch].blocks.keys()))
            architecture_path = f'{RESTORE}/architectures/{_arch}.json'
            if not os.path.exists(architecture_path):
                f = open(architecture_path, 'w')
                json_object = jsonpickle.encode(self.architectures[_arch])
                f.write(json_object)
                f.close()

        message = f'Average blocks :: {sum(average_blocks) / len(average_blocks)} for generation {self.generation}.'
        LOG.info(message)
        # SlackPost.post_neutral('Average blocks', message)

    def restore_architecture(self, key) -> Architecture:
        """
        This method restores an architecture from file.

        :param key: the identifier of the architecture to be restored.
        :return:
        """
        f = open(f'{RESTORE}/architectures/{key}.json')
        json_str = f.read()
        architecture = jsonpickle.decode(json_str)
        f.close()
        return architecture

    def check_if_fitness_default(self, architecture_key):
        """
        This method checks whether the fitness of the provided architecture has a default value.
        This method is only applicable when restoring from a snapshot.
        :param architecture_key:
        :return:
        """
        if architecture_key in self.population_fitness.keys():
            return self.population_fitness[architecture_key].is_ptb_fitness_default()

        return False

    def should_model_be_loaded(self, key):
        if key in self.architectures.keys() or key in self.population_fitness.keys() or key in self.fitness_history.keys():
            return False
        return True

    def test_new_architecture_similarity(self, new_architecture: Architecture) -> bool:
        """
        Compares the provided architecture with all current architectures to determine whether a similar architecture exist.
        :param new_architecture:
        :return:
        """
        for key in self.architectures.keys():
            similarity = new_architecture.compare_with_other(self.architectures[key])
            if similarity == len(new_architecture.blocks.keys()) and len(new_architecture.blocks.keys()) == len(
                    self.architectures[key].blocks.keys()):
                LOG.debug(f'Got a similarity of {similarity} for {key}.')
                return True

            if self.check_if_hash_exist(new_architecture):
                LOG.debug(f'New architecture has same hash as {key}.')
                return True

        return False

    def check_if_hash_exist(self, new_architecture: Architecture) -> bool:
        if len(self.arch_history.keys()) == 0:
            self.sync_arch_history()

        lst_hashes = list(self.arch_history.values())
        _compare_json = json.dumps(new_architecture.get_block_strings())
        _compare_hash = str(hash(_compare_json))
        return _compare_hash in lst_hashes

    def sync_arch_history(self):
        for key in self.architectures.keys():
            if key not in self.arch_history.keys():
                _hash = json.dumps(self.architectures[key].get_block_strings())
                self.arch_history[key] = str(hash(_hash))

    def get_highest_arch_key(self):
        """
        Returns the next identifier to be used for randomly generated architectures during initialisation.
        :return:
        """
        highest = -1
        for existing in self.get_all_keys():
            if existing.find('rdm') > -1:
                num = int(existing[existing.find('m') + 1:existing.find('_')])
                if num > highest:
                    highest = num
        return highest + 1

    def get_all_keys(self):
        """
        Returns all the identifiers that exist across all architectures that exist.
        :return:
        """
        all_keys = set()
        all_keys.update(list(self.architectures.keys()))
        all_keys.update(list(self.population_fitness.keys()))
        all_keys.update(list(self.fitness_history.keys()))
        return all_keys

    def print_current_best_architectures(self):
        """
        This method prints a list of the architectures of the current generation and their performances.
        If posting to Slack is enabled, this method will post a message to Slack to indicate the current best performing architecture.

        :return:
        """
        fitness_values = []
        for f in self.fronts.keys():
            for key in self.fronts[f]:
                if key in self.population_fitness.keys():
                    fitness_values.append(
                        list(self.population_fitness[key].get_fitness_array(self.config('objectives'))) + [
                            key])

        fitness_values = sorted(fitness_values, key=lambda x: (x[0], x[1], x[2]))
        lst = []
        for e, v in enumerate(fitness_values):
            lst.append(f'{e + 1} = {list(v)[-1]};')

        LOG.info(f'Top architectures for {self.generation + 1} :: {" ".join(lst)}')
        SlackPost.post_neutral('Best architecture',
                               f'Current best architecture for {self.generation + 1} is {list(fitness_values[0])[-1]}')

    def get_next_key(self, old_key):
        """
        Generate a new key for an architecture.

        :param old_key:
        :return:
        """
        count = 1
        idx = old_key.find('_') + 1
        new_key = old_key[:idx] + str(count)
        while new_key in self.architectures.keys() or new_key in self.population_fitness.keys() or new_key in self.arch_history.keys():
            count += 1
            new_key = old_key[:idx] + str(count)
        return new_key

    def remove_architecture(self, key):
        """
        This method removes an architecture from the population.

        :param key:
        :return:
        """
        if key in self.architectures.keys():
            self.architecture_transformation_history[key] = self.architectures[key].transformation_history
            self.architectures.pop(key)

        if key in self.population_fitness.keys():
            self.fitness_history[key] = copy.deepcopy(self.population_fitness[key].get_ptb_fitness())

        if key in self.population:
            self.population.remove(key)

        fronts_to_check = []
        for f in self.fronts.keys():
            if key in self.fronts[f]:
                self.fronts[f].remove(key)
                fronts_to_check.append(f)

        for f in fronts_to_check:
            if len(self.fronts[f]) == 0:
                self.fronts.pop(f)
                LOG.info(f'Removed {f} from fronts. {"-" * 72}')

    def sync_model_performances(self):
        updated = []
        for key in self.population_fitness.keys():
            if self.population_fitness[key].is_ptb_fitness_default():
                continue

            if not Persistence.does_model_performance_exist(key):
                if key not in self.architectures.keys():
                    continue

                NASController.set_architecture_performance(key, PtbModelObjectives.PTB_MODEL_LOSS, self.population_fitness[key].ptb_loss)
                NASController.set_architecture_performance(key, PtbModelObjectives.PTB_MODEL_PPL, self.population_fitness[key].ptb_ppl)
                NASController.set_architecture_performance(key, ArchitectureObjectives.NUMBER_OF_BLOCKS, len(self.architectures[key].blocks.keys()))
                NASController.set_architecture_performance(key,
                                                           ArchitectureObjectives.NUMBER_OF_PARAMETERS,
                                                           self.population_fitness[key].number_of_parameters)
                _hash = json.dumps(self.architectures[key].get_block_strings())
                Persistence.persist_model_performance_objectives(key, str(hash(_hash)))
                updated.append(key)

        Persistence.clear_df_cache()
        if len(updated) > 0:
            LOG.debug(f'Persisted performance for: {", ".join(updated)}.')
        else:
            LOG.debug('All current model performances are persisted.')

    def clean_architecture(self, architecture):
        messages = []
        blocks_to_exclude = ['x', 'h', 'c', 'h_next', 'c_next']
        empty_blocks = set()
        for block in architecture.blocks.keys():
            if block in blocks_to_exclude: continue
            if architecture.blocks[block].combination is None and len(architecture.blocks[block].activation) < 1:
                empty_blocks.add(block)

        for key in empty_blocks:
            block = architecture.blocks.pop(key)
            inp = block.inputs[0]
            for blk in architecture.blocks.keys():
                if blk == block:
                    continue

                if key in architecture.blocks[blk].inputs:
                    for i, e in enumerate(architecture.blocks[blk].inputs):
                        if e == key:
                            architecture.blocks[blk].inputs[i] = inp
                    messages.append(f'Updated {blk}')

        return len(empty_blocks)

    def clean_up_architectures(self):
        for arch in self.architectures:
            self.clean_architecture(self.architectures[arch])

    def clean_up_arch_files(self):
        files = glob.glob(f'{RESTORE}/architectures/*_*.json', recursive=False)
        for filename in files:
            f = open(filename, 'r')
            json_str = f.read()
            architecture = jsonpickle.decode(json_str)
            f.close()
            empty_blocks = self.clean_architecture(architecture)
            if empty_blocks > 0:
                f = open(filename, 'w')
                json_object = jsonpickle.encode(architecture)
                f.write(json_object)
                f.close()


    def config(self, key):
        return EnvironmentConfig.get_config(key)

    def check_times(self):
        """
        Check whether the current running time of the search exceed (or attempts to determine whether it will exceed)
        the specified running time limit.
        :return:
        """
        if len(self.generation_times) > 0:
            avg_time = sum(self.generation_times) / len(self.generation_times)
            LOG.info(f'Current average generation training time is {avg_time // 60} minutes.')
            diff = time.time() - self.search_start_time
            terminate_search = -1 < self.config('time_limit') < (diff + avg_time)
            if terminate_search:
                SlackPost.post_neutral('Time constraint',
                                       'Current running time and average generation time exceeds limit set.')
            return terminate_search

        return False

    def log_current_metrics(self):
        """
        Calculates the average loss and perplexity of the current population and logs the metrics.
        :return:
        """

        total_parameters = 0
        total_number_of_blocks = 0
        total_number_of_mul = 0

        ptb_losses = []
        ptb_perplexities = []
        for key in self.architectures.keys():
            if key in self.population_fitness.keys():
                model_fitness = self.population_fitness[key]
                total_parameters += model_fitness.number_of_parameters
                total_number_of_blocks += model_fitness.number_of_blocks
                total_number_of_mul += model_fitness.number_of_mul

                if model_fitness.ptb_loss < 10.0e+7:
                    ptb_losses.append(model_fitness.ptb_loss)

                if model_fitness.ptb_ppl < 10.0e+7:
                    ptb_perplexities.append(model_fitness.ptb_ppl)

        TensorBoardWriter.write_scalar('search', f'Search/average/total_parameters',
                                       total_parameters, self.generation)
        TensorBoardWriter.write_scalar('search', f'Search/average/total_number_of_blocks',
                                       total_number_of_blocks, self.generation)
        TensorBoardWriter.write_scalar('search', f'Search/average/total_number_of_mul',
                                       total_number_of_mul, self.generation)

        update_log = []
        d = 1 if len(ptb_losses) == 0 else len(ptb_losses)
        ptb_loss = sum(ptb_losses) / d
        d_ppl = 1 if len(ptb_perplexities) == 0 else len(ptb_perplexities)
        ptb_ppl = sum(ptb_perplexities) / d_ppl
        TensorBoardWriter.write_scalar('search', f'Search/average/ptb_loss', ptb_loss, self.generation)
        TensorBoardWriter.write_scalar('search', f'Search/average/ptb_ppl', ptb_ppl, self.generation)
        update_log.append(f'{"=" * 10} Penn Treebank dataset {"=" * 10}')
        update_log.append(f'Loss: {ptb_loss}; PPL: {ptb_ppl}.')
        update_log.append(f'{"=" * 10}================{"=" * 10}')

        if ((self.generation + 1) % self.config('post_update_freq')) == 0 and len(update_log) > 0:
            message_to_post = '\n'.join(update_log)
            SlackPost.post_neutral(f'Generation {self.generation + 1} update', message_to_post)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        print('Search delegator instance deleted.')


class TimeExceededException(Exception):
    pass
