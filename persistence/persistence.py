import datetime
import math
import os

import jsonpickle
import pandas as pd
from pymongo import MongoClient
from torch import Tensor

from config.env_config import EnvironmentConfig
from model.architecture_objectives import ArchitectureObjectives
from ops.nas_controller import NASController
from utils.logger import LOG

LOG = LOG.get_instance().get_logger()
CSV_PATH = './output/persistence/tmp.csv'


class Persistence:
    """
    The purpose of this class is only to save architecture metrics, such as number of blocks, perplexity, training time etc.
    """
    __instance = None

    @staticmethod
    def get_instance():
        if Persistence.__instance is None:
            Persistence()
            return Persistence.__instance
        else:
            return Persistence.__instance

    def __init__(self):
        if Persistence.__instance != None:
            raise Exception('Trying to create new instance of existing singleton.')

        self.client = None
        self.existing_performance = []
        self.default_columns = EnvironmentConfig.get_config('default_persistence_columns')
        self.specific_columns = []
        self.update_default_columns()
        self.use_mongo = True
        try:
            self.init_client()
        except:
            self.use_mongo = False
        self.best_achieved = {}
        self.temp_df = None
        self.set_best_performances()
        Persistence.__instance = self

    def update_default_columns(self):
        self.specific_columns = EnvironmentConfig.get_config('ptb_persistence_columns')
        self.default_columns += self.specific_columns
        LOG.info(f'Set default persistence columns : {self.default_columns}')

    def close_connection(self):
        if not EnvironmentConfig.get_config('persist') or not self.use_mongo:
            return
        try:
            LOG.debug('Closing mongo connection.')
            self.client.close()
            del self.client
            LOG.debug('Mongo connection closed.')
        except:
            LOG.debug('Connection might have already been closed.')

    def init_client(self):
        config_instance = EnvironmentConfig.get_instance()
        mongo_user = config_instance.get_config('mongo_user')
        mongo_password = config_instance.get_config('mongo_password')

        if not os.path.exists(CSV_PATH):
            df = pd.DataFrame(columns=self.default_columns)
            df.to_csv(CSV_PATH, index=False)

        try:
            if not config_instance.get_config('persist'):
                LOG.debug('Persistence disabled.')
                return
        except KeyError:
            LOG.debug('Persist enabled.')

        if mongo_user is None or mongo_password is None:
            raise Exception('MongoDB env config not found.')

        connection = f'mongodb+srv://{mongo_user}:{mongo_password}@cluster0.y0kbt.mongodb.net/myFirstDatabase?retryWrites=true&w=majority'

        self.client = MongoClient(connection)
        LOG.info('MongoDB connection established.')

    def set_best_performances(self):
        if not os.path.exists(CSV_PATH):
            return
        df = pd.read_csv(CSV_PATH)
        for idx, row in df.iterrows():
            self.existing_performance.append(row['model_id'])
            for _k in self.specific_columns:
                if _k in row.keys():
                    _val = row[_k]
                    if _k not in self.best_achieved.keys() or _val < self.best_achieved[_k]:
                        self.best_achieved[_k] = _val

        self.load_df_cache()

    def load_df_cache(self):
        self.temp_df = pd.read_csv(CSV_PATH)

    @staticmethod
    def new_entry(type_p, file_p, description_p):
        now = datetime.datetime.now()
        entry = {
            'type': type_p,
            'time': now,
            'file': file_p,
            'description': description_p
        }
        Persistence.get_instance().persist_entry(entry)

    def persist_entry(self, entry_p):
        try:
            if not EnvironmentConfig.get_instance().get_config('persist'):
                LOG.debug('Skipping persistence.')
                return
        except KeyError:
            LOG.debug('Persist enabled.')

        if self.use_mongo and self.client is None:
            raise Exception('MongoDB connection lost.')

        try:
            if entry_p['type'] is None:
                raise Exception('Type must be specified.')
        except KeyError:
            raise Exception('Type must be specified.')

        db = self.client.nas
        entry = db.entry
        entry.insert_one(entry_p)
        LOG.info('Entry persisted.')

    @staticmethod
    def persist_model_performance_objectives(model_identifier, model_hash):
        now = datetime.datetime.now()
        entry = {
            'time': now,
            'model_id': model_identifier,
            'model_hash': model_hash
        }

        persistence_keys = set()
        persistence_keys.add(ArchitectureObjectives.NUMBER_OF_BLOCKS.value)
        persistence_keys.add(ArchitectureObjectives.NUMBER_OF_PARAMETERS.value)
        persistence_keys.add(ArchitectureObjectives.TRAINING_TIME.value)
        persistence_keys.update(Persistence.get_instance().specific_columns)

        for _key in persistence_keys:
            value = NASController.get_architecture_performance_value(model_identifier, _key)
            if type(value) is Tensor:
                value = value.item()
            entry[_key] = value

        Persistence.get_instance().persist_model_performance(entry)

    @staticmethod
    def find_performances():
        return Persistence.get_instance().__find_model_performance()

    def __find_model_performance(self):
        try:
            if not EnvironmentConfig.get_instance().get_config('persist'):
                LOG.debug('Persistence disabled.')
                return
        except KeyError:
            LOG.debug('Persist enabled.')

        if self.use_mongo:
            db = self.client.nas
            performance = db.performance
            performances = performance.find().sort("ptb_ppl")
            return list(performances)
        elif os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            rows = []
            for idx, row in df.iterrows():
                rows.append((row['ptb_ppl'], row['model_id']))
            rows = sorted(rows, key=lambda x: x[0])
            performances = []
            for r in rows:
                ppl, id = r
                performances.append({'ptb_ppl': ppl, 'model_id': id})
            return performances
        return []

    def persist_model_performance(self, entry):
        try:
            if not EnvironmentConfig.get_instance().get_config('persist'):
                LOG.debug('Skipping persistence.')
                return
        except KeyError:
            LOG.debug('Persist enabled.')

        try:
            if entry['model_id'] is None:
                raise Exception('Model ID must be specified.')
        except KeyError:
            raise Exception('Model ID must be specified.')

        if self.use_mongo and self.client is not None:
            db = self.client.nas
            performance = db.performance
            query = {"model_id": entry['model_id']}
            result = performance.find(query)
            if len(list(result)) > 0:
                LOG.info(f'Entry for {entry["model_id"]} already exist.')
                return
            else:
                performance.insert_one(entry)
        else:
            self.save_to_csv(entry)
        LOG.info(f'Performance for {entry["model_id"]} persisted.')

    def save_to_csv(self, entry):
        if not os.path.exists('./output/persistence'):
            os.mkdir('./output/persistence')

        if os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
        else:
            df = pd.DataFrame(columns=self.default_columns)

        if len(list(df.loc[df['model_id'] == entry['model_id']]['model_id'])) > 0:

            persistence_keys = set()
            persistence_keys.add(ArchitectureObjectives.NUMBER_OF_BLOCKS.value)
            persistence_keys.add(ArchitectureObjectives.NUMBER_OF_PARAMETERS.value)
            persistence_keys.add(ArchitectureObjectives.TRAINING_TIME.value)
            persistence_keys.update(Persistence.get_instance().specific_columns)

            LOG.debug(f'Entry for {entry["model_id"]} already exist.')

            found = False
            for index in df.index:
                if found:
                    break
                if df.loc[index, 'model_id'] == entry['model_id']:
                    for clm in persistence_keys:
                        df.loc[index, clm] = entry[clm]
                    found = True

        else:
            df = df.append(entry, ignore_index=True)

        df = df.sort_values(by=self.specific_columns, ascending=True)
        df.to_csv(CSV_PATH, index=False)

    @staticmethod
    def set_current_state(generation, model_identifier):
        state = Persistence.get_instance().find_current_state()
        now = datetime.datetime.now()

        db = Persistence.get_instance().client.nas
        col = db.state
        if state is None:
            state = {
                'time': now,
                'generation': generation,
                'model': model_identifier
            }
            col.insert_one(state)
        else:
            new_values = {'$set': {'time': now,
                                   'generation': generation,
                                   'model': model_identifier}}
            col.update_one(state, new_values)

    def find_current_state(self):
        db = self.client.nas
        state = db.state
        return state.find_one()

    @staticmethod
    def is_new_best(model_fitness):
        result = []
        for key in Persistence.get_instance().specific_columns:
            value = getattr(model_fitness, key)
            if key not in Persistence.get_instance().best_achieved.keys() or value < \
                    Persistence.get_instance().best_achieved[key]:
                Persistence.get_instance().best_achieved[key] = value
                result.append(key)
        return result

    @staticmethod
    def get_current_best(key):
        if key in Persistence.get_instance().best_achieved.keys():
            return Persistence.get_instance().best_achieved[key]
        return None

    @staticmethod
    def does_model_performance_exist(model_identifier):
        persistence_keys = set()
        persistence_keys.add(ArchitectureObjectives.NUMBER_OF_BLOCKS.value)
        persistence_keys.add(ArchitectureObjectives.NUMBER_OF_PARAMETERS.value)
        persistence_keys.add(ArchitectureObjectives.TRAINING_TIME.value)
        persistence_keys.update(Persistence.get_instance().specific_columns)

        if Persistence.get_instance().temp_df is None:
            Persistence.get_instance().load_df_cache()

        df = Persistence.get_instance().temp_df
        row = df.loc[df['model_id'] == model_identifier]

        if row is None or row.empty:
            return False

        for _key in persistence_keys:
            if row[_key].values[0] is None or math.isnan(row[_key].values[0]):
                return False

        return True

    def sync_with_mongo(self):
        if not os.path.exists(CSV_PATH):
            raise Exception('CSV does not exist.')

        if not self.use_mongo or self.client is None:
            raise Exception('Could not establish a connection to MongoDB.')

        df = pd.read_csv(CSV_PATH)

        fixed = False
        for col in df.columns:
            if col.find('Unnamed') == 0:
                df.drop(col, axis=1, inplace=True)
                fixed = True
        if fixed:
            df.to_csv(CSV_PATH, index=False)

        entries = {}
        for idx, row in df.iterrows():
            if row['model_id'] not in self.existing_performance:
                entries[row['model_id']] = {
                    'time': row['time'],
                    'model_id': row['model_id'],
                    'ptb_loss': row['ptb_loss'],
                    'ptb_ppl': row['ptb_ppl'],
                    'training_time': row['training_time'],
                    'model_hash': row['model_hash']
                }

        LOG.info(f'Syncing models :: {", ".join(list(entries.keys()))}')
        for key in entries.keys():
            self.persist_model_performance(entries[key])

    @staticmethod
    def sync_csv_to_db():
        Persistence.get_instance().sync_with_mongo()
        LOG.info('Performance synced.')

    def load_persistence_dataframe(self):
        if not os.path.exists(CSV_PATH):
            raise Exception('CSV does not exist.')

        df = pd.read_csv(CSV_PATH, index_col='model_id')
        return df

    def drop_collection(self):
        db = self.client.nas
        performance = db.performance
        LOG.info('Dropping collection, hope you made a backup.')
        performance.drop()
        LOG.info('Collection dropped.')

    @staticmethod
    def persist_architecture(architecture):
        if not os.path.exists('./restore/architectures'):
            os.mkdir('./restore/architectures')

        if not os.path.exists(f'./restore/architectures/{architecture.identifier}.json'):
            f = open(f'./restore/architectures/{architecture.identifier}.json', 'w')
            json_object = jsonpickle.encode(architecture)
            f.write(json_object)
            f.close()

    @staticmethod
    def clear_df_cache():
        Persistence.get_instance().temp_df = None
