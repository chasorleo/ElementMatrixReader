import numpy as np


def lines_from_file(textfile):
    with open(textfile, 'rt', encoding='utf8') as flines:
        yield from stripped_lines(flines)


def stripped_lines(lines):
    for line in lines:
        if line and not line.startswith('*'):
            yield line.strip()


class ElementMatrixReader:
    def __init__(self, job):
        self.job = job
        # mtx generated with
        #   *MATRIX GENERATE,STIFFNESS,LOAD,ELEMENT BY ELEMENT
        #   *MATRIX OUTPUT,FORMAT=MATRIX INPUT,STIFFNESS,LOAD
        self.element_stiffness = job + '_STIF1.mtx'
        self.element_load = job + '_LOAD1.mtx'
        # mtx generated with
        #   *MATRIX GENERATE,STIFFNESS,LOAD
        #   *MATRIX OUTPUT,FORMAT=COORDINATE,STIFFNESS,LOAD
        self.global_stiffness = job + '_STIF2.mtx'
        self.global_load = job + '_LOAD2.mtx'

        self.dof = self.get_dof_per_node()
        self.element = self.get_element_labels()
        self.node = self.get_node_labels()
        self.total_dof = self.get_total_dofs()
        self.Ki = self.get_element_stiffness_matrix()
        self.fi = self.get_element_load()
        self.K = self.get_global_stiffness_matrix()
        self.f = self.get_global_load_vector()

    def get_dof_per_node(self):
        dof = -1
        for l in lines_from_file(self.element_stiffness):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            if row_dof > dof:
                dof = row_dof
        return dof

    def get_node_labels(self):
        node = set()
        for l in lines_from_file(self.element_stiffness):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            node.add(row_node)
            node.add(col_node)
        return node

    def get_element_labels(self):
        element = {}
        for l in lines_from_file(self.element_stiffness):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            elemend_nodes = (row_node, col_node)
            if elemend_nodes[0] != elemend_nodes[1]:
                element.update({element_id: elemend_nodes})
        return element

    def get_element_stiffness_matrix(self):
        Kis = {}
        for ele_id in self.element:
            Kis.update({ele_id: np.zeros((2 * self.dof, 2 * self.dof))})

        for l in lines_from_file(self.element_stiffness):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            matrix_entry = float(l.split(',')[-1])
            row = (self.element[element_id].index(row_node) - 1) * self.dof + row_dof - 1
            col = (self.element[element_id].index(col_node) - 1) * self.dof + col_dof - 1
            Kis[element_id][row, col] = matrix_entry
            Kis[element_id][col, row] = matrix_entry
        return Kis

    def get_element_load(self):
        Fis = {}
        for ele_id in self.element:
            Fis.update({ele_id: np.zeros((2 * self.dof, 1))})

        for l in lines_from_file(self.element_load):
            element_id, node_id, row_dof, x = list(map(int, l.split(',')[:-1]))
            rhs_vector_entry = float(l.split(',')[-1])
            row = (self.element[element_id].index(node_id) - 1) * self.dof + row_dof - 1
            Fis[element_id][row] = rhs_vector_entry
        return Fis

    def get_total_dofs(self):
        total_dof = -1
        for l in lines_from_file(self.global_stiffness):
            row = int(l.split()[0])
            if total_dof < row:
                total_dof = row
        return total_dof

    def get_global_stiffness_matrix(self):
        K = np.zeros((self.total_dof, self.total_dof))

        for l in lines_from_file(self.global_stiffness):
            row, col = list(map(int, l.split()[:-1]))
            matrix_entry = float(l.split()[-1])
            K[row - 1, col - 1] = matrix_entry
            K[col - 1, row - 1] = matrix_entry
        return K

    def get_global_load_vector(self):
        F = np.zeros((self.total_dof, 1))

        for l in lines_from_file(self.global_load):
            row = list(map(int, l.split()[:-1]))[0]
            rhs_vector = float(l.split()[-1])
            F[row-1] = rhs_vector
        return F


if __name__ == "__main__":
    job = 'truss'
    model = ElementMatrixReader(job)

    print(f"Model Summary:")
    print(f"Degree of freedom per node: {model.dof}")
    print(f"Node list:\n   {model.node}")
    print(f"element list (element_id: (node_1, node_2)):\n   {model.element}")

    for i in model.element:
        print(f"ELEMENT STIFFNESS MATRIX OF ELEMENT: {i},  k_{i} = ")
        print(model.Ki[i])

    print("GLOBAL STIFFNESS MATRIX (K): ")
    print(model.K)
    print("GLOBAL LOAD VECTOR(f): ")
    print(model.f)


