import copy
import json
import os


class EnvironmentConfig:
    __instance = None

    def __init__(self):
        if EnvironmentConfig.__instance is not None:
            raise Exception('Trying to create new instance of existing singleton.')

        if not os.path.exists('./config'):
            os.mkdir('./config')

        if not os.path.exists('./config/env.json'):
            self.initialise_default_config()

        self.load_config()

        try:
            self.post_slack = self.config['slack_webhook'] != -1
        except KeyError:
            self.post_slack = False

        EnvironmentConfig.__instance = self

    @staticmethod
    def get_instance():
        if EnvironmentConfig.__instance is None:
            EnvironmentConfig()
            return EnvironmentConfig.__instance
        else:
            return EnvironmentConfig.__instance

    def load_config(self):
        with open('config/env.json') as file:
            self.config = json.load(file)

        try:
            if os.path.exists('C:\\dev\\local.txt'):
                self.is_local_env = True
            else:
                self.is_local_env = False
        except:
            self.is_local_env = True

    def get_config_loaded(self, key):
        if key in self.config.keys():
            return copy.deepcopy(self.config[key])
        return None

    def update_post_slack(self, value: bool):
        self.post_slack = value

    @staticmethod
    def get_config(key):
        return EnvironmentConfig.get_instance().get_config_loaded(key)

    @staticmethod
    def get_post_slack():
        return EnvironmentConfig.get_instance().post_slack

    @staticmethod
    def get_slack_webhook():
        try:
            if 'slack_webhook' in EnvironmentConfig.get_instance().config.keys():
                return EnvironmentConfig.get_instance().config['slack_webhook']
        except KeyError:
            return None
        return None

    def initialise_default_config(self):
        default = {
            "env_name": "DEFAULT",
            "persist": True,
            "slack_webhook": -1,
            "post_update_freq": 1,
            "time_limit": -1,
            "restore_if_possible": True,
            "logging_debug": False,

            "population_size": 100,
            "population_to_select": 100,
            "number_of_generations": 25,
            "override_transformation_count": 3,
            "alternative_nsga_crowding": False,
            "initial_population_transformations_min": 1,
            "initial_population_transformations": 10,

            "warm_start_models": True,
            "include_lstm": True,
            "include_gru": True,

            "character_level": True,
            "dataset": "ptb",

            "ptb_training_epochs": 15,
            "ptb_ppl_upper_threshold": 500000,
            "ptb_model_nlayers": 1,
            "force_ptb_training": True,
            "ptb_compare_ppl_threshold": 5,
            "ptb_threshold_epochs": 5,
            "override_train_data": False,

            "objectives": [
                "number_of_blocks",
                "ptb_ppl"
            ],
            "cheap_objectives": [
                "number_of_parameters",
                "number_of_blocks"
            ],

            "default_persistence_columns": [
                "time", "model_id", "model_hash", "number_of_blocks", "number_of_parameters", "training_time"
            ],

            "ptb_persistence_columns": [
                "ptb_ppl", "ptb_loss"
            ],

            "activation_functions": [
                "identity",
                "tanh",
                "sigmoid",
                "relu",
                "leaky_relu",
                "linear",
                "linear_b",
                "None"
            ],
            "combination_methods": [
                "add",
                "elem_mul",
                "sub"
            ],
            "network_transformations": [
                "add_unit",
                "remove_unit",
                "add_connection",
                "remove_connection",
                "add_recurrent_connection",
                "change_random_activation",
                "change_combination_connection"
            ],

            "identity": 0.1,
            "tanh": 0.1,
            "sigmoid": 0.1,
            "relu": 0.1,
            "linear": 0.1,
            "linear_b": 0.1,
            "None": 0.1,
            "add": 0.1,
            "elem_mul": 0.1,
            "sub": 0.1,

            "add_unit": 0.2,
            "remove_unit": 0.2,
            "add_gate": 0.2,
            "remove_gate": 0.02,
            "add_connection": 0.2,
            "remove_connection": 0.2,
            "add_recurrent_connection": 0.2,
            "remove_recurrent_connection": 0.2,
            "change_transfer_function": 0.2,
            "change_combination_connection": 0.2,
            "flip_network_type": 0.001,
            "flip_time_delay": 0.25,
            "flip_input": 0.15,
            "flip_output": 0.15,
            "clean_up": 0.5
        }

        f = open('./config/env.json', 'w')
        json_object = json.dumps(default)
        f.write(json_object)
        f.close()
