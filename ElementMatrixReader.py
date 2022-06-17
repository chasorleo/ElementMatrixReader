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
        self.element_stiffness_M = job + '_STIF1.mtx'
        self.element_load_M = job + '_LOAD1.mtx'
        # mtx generated with
        #   *MATRIX GENERATE,STIFFNESS,LOAD,ELEMENT BY ELEMENT
        #   *MATRIX OUTPUT,FORMAT=COORDINATE,STIFFNESS,LOAD
        self.element_stiffness_C = job + '_STIF2.mtx'
        self.element_load_C = job + '_LOAD2.mtx'
        # mtx generated with
        #   *MATRIX GENERATE,STIFFNESS,LOAD
        #   *MATRIX OUTPUT,FORMAT=COORDINATE,STIFFNESS,LOAD
        self.global_stiffness_C = job + '_STIF4.mtx'
        self.global_load_C = job + '_LOAD4.mtx'
        # mtx generated with
        #   *MATRIX GENERATE,STIFFNESS,LOAD
        #   *MATRIX OUTPUT,FORMAT=MATRIX INPUT,STIFFNESS,LOAD
        self.global_stiffness_M = job + '_STIF3.mtx'
        self.global_load_M = job + '_LOAD3.mtx'

        self.dof = self.get_dof_per_node()
        self.element = self.get_element_labels()
        self.node = self.get_node_labels()
        self.total_dof = self.get_total_dofs()
        # element stiffness matrix compact
        self.ki = self.get_element_stiffness_matrix()
        # element stiffness matrix expand to global stiffness dimension
        self.Ki = self.get_expanded_element_stiffness_matrix()
        self.fi = self.get_element_load()
        self.K = self.get_global_stiffness_matrix()
        self.f = self.get_global_load_vector()
        self.KK = self.get_globel_stiffness_with_expaned_element_stiffness_matrix()

    def get_dof_per_node(self):
        dof = -1
        for l in lines_from_file(self.element_stiffness_M):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            if row_dof > dof:
                dof = row_dof
        return dof

    def get_node_labels(self):
        node = set()
        for l in lines_from_file(self.element_stiffness_M):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            node.add(row_node)
            node.add(col_node)
        return node

    def get_element_labels(self):
        element = {}
        for l in lines_from_file(self.element_stiffness_M):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            elemend_nodes = (row_node, col_node)
            if elemend_nodes[0] != elemend_nodes[1]:
                element.update({element_id: elemend_nodes})
        return element

    def get_element_stiffness_matrix(self):
        kis = {}
        for ele_id in self.element:
            kis.update({ele_id: np.zeros((2 * self.dof, 2 * self.dof))})

        for l in lines_from_file(self.element_stiffness_M):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            matrix_entry = float(l.split(',')[-1])
            row = (self.element[element_id].index(row_node) - 1) * self.dof + row_dof - 1
            col = (self.element[element_id].index(col_node) - 1) * self.dof + col_dof - 1
            kis[element_id][row, col] = matrix_entry
            kis[element_id][col, row] = matrix_entry

        for l in lines_from_file(self.element_stiffness_C):
            element_id, row, col = list(map(int, l.split()[:-1]))
            matrix_entry = float(l.split()[-1])
            element_id += 1
            row -= 1
            col -= 1
            np.isclose(kis[element_id][row, col], matrix_entry)

        return kis

    def get_expanded_element_stiffness_matrix(self):
        Kis = {}
        for ele_id in self.element:
            Kis.update({ele_id: np.zeros((self.total_dof, self.total_dof))})

        for l in lines_from_file(self.element_stiffness_M):
            element_id, row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            matrix_entry = float(l.split(',')[-1])
            row = (row_node - 1) * self.dof + row_dof - 1
            col = (col_node - 1) * self.dof + col_dof - 1
            Kis[element_id][row, col] = matrix_entry
            Kis[element_id][col, row] = matrix_entry
        return Kis

    def get_element_load(self):
        Fis = {}
        for ele_id in self.element:
            Fis.update({ele_id: np.zeros((2 * self.dof, 1))})

        for l in lines_from_file(self.element_load_M):
            element_id, node_id, row_dof, x = list(map(int, l.split(',')[:-1]))
            rhs_vector_entry = float(l.split(',')[-1])
            row = (self.element[element_id].index(node_id) - 1) * self.dof + row_dof - 1
            Fis[element_id][row] = rhs_vector_entry

        for l in lines_from_file(self.element_load_C):
            element_id, row, x = list(map(int, l.split()[:-1]))
            matrix_entry = float(l.split()[-1])
            element_id += 1
            row -= 1
            np.isclose(Fis[element_id][row], matrix_entry)

        return Fis

    def get_total_dofs(self):
        total_dof = -1
        for l in lines_from_file(self.global_stiffness_C):
            row = int(l.split()[0])
            if total_dof < row:
                total_dof = row
        return total_dof

    def get_global_stiffness_matrix(self):
        K = np.zeros((self.total_dof, self.total_dof))

        for l in lines_from_file(self.global_stiffness_C):
            row, col = list(map(int, l.split()[:-1]))
            matrix_entry = float(l.split()[-1])
            K[row - 1, col - 1] = matrix_entry
            K[col - 1, row - 1] = matrix_entry

        # double check with MATRIX INPUT format output
        for l in lines_from_file(self.global_stiffness_M):
            row_node, row_dof, col_node, col_dof = list(map(int, l.split(',')[:-1]))
            matrix_entry = float(l.split()[-1])
            row = (row_node - 1) * 2 + row_dof - 1
            col = (col_node - 1) * 2 + col_dof - 1
            assert np.isclose(K[row, col], matrix_entry)
        return K

    def get_global_load_vector(self):
        F = np.zeros((self.total_dof, 1))

        for l in lines_from_file(self.global_load_C):
            row = list(map(int, l.split()[:-1]))[0]
            rhs_vector = float(l.split()[-1])
            F[row - 1] = rhs_vector

        # double check with MATRIX INPUT format output
        for l in lines_from_file(self.global_load_M):
            row_node, row_dof = list(map(int, l.split(',')[:-1]))
            matrix_entry = float(l.split()[-1])
            row = (row_node - 1) * 2 + row_dof - 1
            assert np.isclose(F[row], matrix_entry)

        return F

    def get_globel_stiffness_with_expaned_element_stiffness_matrix(self):
        K = np.zeros((self.total_dof, self.total_dof))
        for ele_id in self.element:
            K += self.Ki[ele_id]
        return K

    def test_globel_stiffness_with_expaned_element_stiffness_matrix(self):
        for i in range(self.total_dof):
            for j in range(self.total_dof):
                if self.K[i, j] < 1e36:
                    assert np.isclose(self.KK[i, j], self.K[i, j])

        return True

if __name__ == "__main__":
    job = 'truss'
    model = ElementMatrixReader(job)
    if model.test_globel_stiffness_with_expaned_element_stiffness_matrix():
        print(f"Model Summary:")
        print(f"Degree of freedom per node: {model.dof}")
        print(f"Node list:\n   {model.node}")
        print(f"element list (element_id: (node_1, node_2)):\n   {model.element}")

        for i in model.element:
            print(f"ELEMENT STIFFNESS MATRIX OF ELEMENT: {i},  k_{i} = ")
            print(model.ki[i])
            print(f"ELEMENT STIFFNESS MATRIX OF ELEMENT: {i},  k_{i} = ")
            print(model.Ki[i])

        print("GLOBAL LOAD VECTOR(f): ")
        print(model.f)
        print("GLOBAL STIFFNESS MATRIX (K): ")
        print(model.K)
        print(model.KK)
