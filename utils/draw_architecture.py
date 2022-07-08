import asyncio
import os

import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from networkx.drawing.nx_pydot import graphviz_layout
from pandas.plotting import table

from model.architecture import Architecture
from persistence.persistence import Persistence

sns.set_theme()


class DrawArchitecture:
    """
    Create images to visualise the architectures.
    """

    def __init__(self, model_identifier, generation=0):
        self.model_identifier = model_identifier
        self.generation = generation

    def save_tree(self, tree, labels):
        GREEN = '#5fad56'
        YELLOW = '#f2c14e'
        PLATINUM = '#e2e2e2'
        GRAY = '#b2abbf'
        BLUE = '#52d1dc'
        RED = '#ff6663'

        if not tree.is_directed():
            print(f'WARN: {self.model_identifier} is not directed.')

        nx.set_node_attributes(tree, labels, 'label')
        p = nx.drawing.nx_pydot.to_pydot(tree)

        for edge in p.get_edges():
            edge.set('fillcolor', RED)
            edge.set('edge_color', RED)

        for node in p.get_nodes():
            node.set('style', 'filled')
            if labels[int(node.get_name())] in ['x', 'h', 'c']:
                node.set('fillcolor', GREEN)
            elif labels[int(node.get_name())] in ['tanh', 'sigmoid', 'identity', 'linear', 'linear_b', 'relu']:
                node.set('fillcolor', YELLOW)
            elif labels[int(node.get_name())] in ['add', 'elem_mul', 'sub']:
                node.set('fillcolor', PLATINUM)
            elif labels[int(node.get_name())] in ['h_next', 'c_next']:
                node.set('fillcolor', BLUE)
            elif type(labels[int(node.get_name())]) == int:
                node.set('fillcolor', RED)
            else:
                node.set('fillcolor', GRAY)

        if not os.path.exists(f'./output/architectures/{self.generation}'):
            os.makedirs(f'./output/architectures/{self.generation}')
        p.write_png(f'./output/architectures/{self.generation}/{self.model_identifier}.png')

    def draw(self, architecture: Architecture, new_identifier=None):
        if new_identifier is not None:
            self.model_identifier = new_identifier

        graph = nx.DiGraph()

        node_dictionary = {}
        # input nodes
        node_dictionary[0] = 'x'
        node_dictionary[1] = 'h'
        node_dictionary[2] = 'c'

        # output nodes
        node_dictionary[3] = 'h_next'
        node_dictionary[4] = 'c_next'

        edges = []
        subs = {}
        remove_keys = set()

        function_keys = {}
        function_inps = {}

        # output nodes
        function_inps['h_next'] = 0
        function_inps['c_next'] = 1

        for block_key in architecture.blocks.keys():
            block = architecture.blocks[block_key]
            if block.combination is None and len(block.activation) == 0:
                if block.identifier is not None and block.identifier not in node_dictionary.values():
                    output_idx = len(node_dictionary.keys())
                    node_dictionary[output_idx] = block.identifier
                else:
                    output_idx = list(node_dictionary.values()).index(block.identifier)

                function_inps[block.identifier] = output_idx
                function_keys[block.identifier] = output_idx

            elif block.combination is not None and len(block.activation) == 0:
                inp_edge_idx = len(node_dictionary.keys())
                node_dictionary[inp_edge_idx] = block.combination

                function_inps[block.identifier] = inp_edge_idx
                function_keys[block.identifier] = inp_edge_idx

            elif len(block.activation) > 0 and block.combination is None:
                inp_edge_idx = len(node_dictionary.keys())
                node_dictionary[inp_edge_idx] = block.activation[0]

                function_inps[block.identifier] = inp_edge_idx
                function_keys[block.identifier] = inp_edge_idx

                subs[block.identifier] = inp_edge_idx

        for block_key in architecture.blocks.keys():
            block = architecture.blocks[block_key]
            for inp in block.inputs:
                if inp not in function_keys.keys():
                    if inp not in node_dictionary.values():
                        inp_idx = len(node_dictionary.keys())
                        node_dictionary[inp_idx] = inp
                    else:
                        inp_idx = list(node_dictionary.values()).index(inp)
                else:
                    inp_idx = function_keys[inp]

                graph.add_edge(inp_idx, function_inps[block.identifier])
                edges.append((inp_idx, function_inps[block.identifier]))

        for k in remove_keys:
            # node_dictionary.pop(k)
            print(f'{k}')
        graph.add_nodes_from(node_dictionary.keys())
        self.save_tree(graph, node_dictionary)

    def save_model(self, graph, labels):
        color_map = []

        GREEN = '#5fad56'
        YELLOW = '#f2c14e'
        PLATINUM = '#e2e2e2'
        GRAY = '#b2abbf'
        BLUE = '#52d1dc'

        def cm_to_inch(value):
            return value / 2.54

        for node in graph:
            if labels[node] in ['x', 'h', 'c_t']:
                color_map.append(GREEN)
            elif labels[node] in ['tanh', 'sigmoid', 'identity', 'linear', 'linear_b']:
                color_map.append(YELLOW)
            elif labels[node] in ['add', 'elem_mul', 'sub']:
                color_map.append(PLATINUM)
            elif labels[node] in ['h_next', 'c_next']:
                color_map.append(BLUE)
            else:
                color_map.append(GRAY)

        plt.figure(figsize=(cm_to_inch(50), cm_to_inch(50)))
        # nx.set_node_attributes(graph, labels, 'label')
        # pos = nx.spring_layout(graph)
        pos = graphviz_layout(graph, prog="dot")
        # nx.draw_networkx(graph, pos=pos, arrows=True, node_size=1500, node_color=color_map, width=1.8, with_labels=False) #node_color='#fcba03'
        nx.draw_networkx(graph, pos=pos, arrows=True, node_size=1500, node_color=color_map, width=1.8,
                         with_labels=False)  # node_color='#fcba03'

        nx.draw_networkx_labels(graph, pos, labels)
        plt.savefig(f'output/graph_{self.model_identifier}.png')

    async def draw_async(self, architecture: Architecture):
        print(f'Draw async {architecture.identifier}.')
        self.draw(architecture, architecture.identifier)

    async def delegate_drawing(self, architectures):
        for arch in architectures:
            asyncio.ensure_future(self.draw_async(arch))

    def draw_architectures(self, architectures):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.delegate_drawing(architectures))
        print('Waiting')
        pending = asyncio.Task.all_tasks()
        loop.run_until_complete(asyncio.gather(*pending))

    def test_dag(self, architecture: Architecture):
        graph = nx.DiGraph()

        for block in architecture.blocks.keys():
            graph.add_node(block)

        for block in architecture.blocks.keys():
            for inp in architecture.blocks[block].inputs:
                graph.add_edge(block, inp)

        print(graph.is_directed())
        print('')

    def draw_performance_heatmap(self):
        df = Persistence.get_instance().load_persistence_dataframe()
        df = df.drop('model_hash', 1)
        df = df.drop('time', 1)
        specifics = Persistence.get_instance().specific_columns

        df = df.sort_values(by=[specifics[0]], ascending=True)
        for _key in specifics:
            df[_key] = df[_key].apply(lambda x: round(x, 4))
            df[_key] = df[_key].apply(lambda x: x if x < 10.0e+2 else "{:.2e}".format(x))
        df['training_time'] = df['training_time'].apply(lambda x: round(x, 4))

        fig, ax = plt.subplots(figsize=(12, 9))  # set size frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
        tabla = table(ax, df, loc='upper right', colWidths=[0.17] * len(df.columns))  # where df is your data frame
        tabla.auto_set_font_size(False)  # Activate set fontsize manually
        tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
        tabla.scale(1.2, 1.2)  # change size table
        plt.savefig('./output/table.png', transparent=True)
