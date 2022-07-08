from model.block import Block


class Architecture(object):
    """
    This is the main Architecture class.

    Network transformations are performed to this class.
    """

    def __init__(self, identifier=None):
        self.identifier = identifier
        self.blocks = {}
        self.generation = 0
        self.fitness = None
        self.transformation_history = []
        self.removed_blocks = []
        self.ancestors = []
        self.main_parent = None
        self.elman_network = True
        self.output_layer_keys = ['h_next', 'c_next']

    def add_block(self, block: Block):
        if block.identifier in self.blocks.keys():
            print(f'Overwriting existing block at {block.identifier}.')
        self.blocks[block.identifier] = block

    def update_generation(self):
        self.generation += 1

    def get_all_block_keys(self):
        return list(self.blocks.keys())

    def clear_block_output_dimensions(self):
        for key in self.blocks.keys():
            self.blocks[key].output_dimension = None

    def get_block_strings(self):
        block_strings = {}

        for key in self.blocks.keys():
            block = self.blocks[key]
            for inp in block.inputs:
                if type(inp) is not int:
                    if inp not in block_strings.keys():
                        block_strings[inp] = self.blocks[inp].get_str()
            if key not in block_strings.keys():
                block_strings[key] = block.get_str()

        return block_strings

    def verify_blocks(self):
        for output in self.output_layer_keys:
            inputs = self.blocks[output].inputs
            for input in inputs:
                encountered = {}
                if type(input) is not int:
                    self.blocks[input].validate(self, encountered)

    def compare_with_other(self, other) -> int:
        block_strings = []
        for key in self.blocks.keys():
            block_strings.append(self.blocks[key].get_str())

        similarity_count = 0
        for key in other.blocks.keys():
            other_block = other.blocks[key].get_str()
            if other_block in block_strings:
                similarity_count += 1

        return similarity_count

    def get_parent(self):
        if len(self.ancestors) < 1:
            return None
        return self.ancestors[-1]

    def get_blocks_that_use(self, input_key):
        blocks_that_use = set()
        for key in self.blocks.keys():
            if key != input_key:
                if input_key in self.blocks[key].inputs:
                    blocks_that_use.add(key)

        return list(blocks_that_use)

    def search_graph(self, visited, block):
        if block not in visited:
            visited.add(block)
            for inp in self.blocks[block].inputs:
                self.search_graph(visited, inp)
