import unittest

from gyraph.utils import load, op, np
from gyraph.graphs import Graph

from gyraph.filters import PolynomialFilter, DualPolynomialFilter


class TestPolynomialFilterFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures for polynomial filter tests.
        """
        self.test_graphs_path = "./tests/test_graphs/"

        self.A = load(op.join(self.test_graphs_path, "bretagne_graph.pkl"))["struct"]
        self.graph = Graph(adj_matrix=self.A)
        self.graph.set_operator("advection_diffusion")

        # Instantiate only to verify no errors in initialization
        self.order = 4
        self.graph_filter_init = PolynomialFilter(graph=self.graph)
        self.graph_filter = PolynomialFilter(graph=self.graph, order=self.order)

        self.dual_graph_filter_dicts = {}
        for filter_type in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            self.dual_graph_filter_dicts[filter_type] = DualPolynomialFilter(
                graph=self.graph, order=self.order, filter_type=filter_type
            )

        self.assertRaises(
            ValueError,
            DualPolynomialFilter,
            graph=self.graph,
            order=self.order,
            filter_type="giberish",
        )

        self.signal = np.arange(self.graph.N)
        self.kernel = np.ones(self.graph.N)

        self.dim = 3

    def test_apply(self):
        """
        Test the application of the polynomial filter.
        """
        filtered = self.graph_filter.apply(self.signal, self.kernel)
        filtered, coefs = self.graph_filter.apply(
            self.signal, self.kernel, return_coefs=True
        )
        self.assertEqual(filtered.shape, self.signal.shape)
        self.assertEqual(len(coefs), self.graph_filter.params["order"])

        for filter_type in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            dual_filtered = self.dual_graph_filter_dicts[filter_type].apply(
                self.signal, self.kernel
            )
            dual_filtered, coefs = self.dual_graph_filter_dicts[filter_type].apply(
                self.signal, self.kernel, return_coefs=True
            )
            self.assertEqual(dual_filtered.shape, self.signal.shape)
            if filter_type == "GAGD":
                self.assertEqual(
                    len(coefs),
                    self.dual_graph_filter_dicts[filter_type].params["order"] * 2,
                )
            else:
                self.assertEqual(
                    len(coefs),
                    self.dual_graph_filter_dicts[filter_type].params["order"],
                )

    def test_polynomial_filter(self):
        """
        Test polynomial filter creation and coefficient retrieval.
        """
        # Instantiate only to verify no errors in initialization
        poly_filter = self.graph_filter.polynomial_filter(
            self.kernel, return_coefs=False
        )
        poly_filter, coefs = self.graph_filter.polynomial_filter(
            self.kernel, return_coefs=True
        )
        self.assertEqual(poly_filter.shape[0], self.signal.shape[0])
        self.assertEqual(len(coefs), self.graph_filter.params["order"])

        for filter_type in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            dual_filter = self.dual_graph_filter_dicts[filter_type].polynomial_filter(
                self.kernel
            )
            dual_filter, coefs = self.dual_graph_filter_dicts[
                filter_type
            ].polynomial_filter(self.kernel, return_coefs=True)
            self.assertEqual(dual_filter.shape[0], self.signal.shape[0])
            if filter_type == "GAGD":
                self.assertEqual(
                    len(coefs),
                    self.dual_graph_filter_dicts[filter_type].params["order"] * 2,
                )
            else:
                self.assertEqual(
                    len(coefs),
                    self.dual_graph_filter_dicts[filter_type].params["order"],
                )

    def test_precompute_polynomial(self):
        """
        Test precomputation of polynomial powers.
        """
        self.graph_filter.precompute_polynomial()
        self.assertEqual(
            len(self.graph_filter.powers_of_M), self.graph_filter.params["order"]
        )

        self.dual_graph_filter_dicts["GAGD"].precompute_polynomial()
        self.assertEqual(
            len(self.dual_graph_filter_dicts["GAGD"].powers_of_P),
            self.dual_graph_filter_dicts["GAGD"].params["order"],
        )
        self.assertEqual(
            len(self.dual_graph_filter_dicts["GAGD"].powers_of_Q),
            self.dual_graph_filter_dicts["GAGD"].params["order"],
        )

        self.dual_graph_filter_dicts["GQAD"].precompute_polynomial()
        self.assertEqual(
            len(self.dual_graph_filter_dicts["GQAD"].powers_of_R),
            self.dual_graph_filter_dicts["GQAD"].params["order"],
        )

        self.dual_graph_filter_dicts["GQDA"].precompute_polynomial()
        self.assertEqual(
            len(self.dual_graph_filter_dicts["GQDA"].powers_of_R),
            self.dual_graph_filter_dicts["GQDA"].params["order"],
        )

        self.dual_graph_filter_dicts["GA"].precompute_polynomial()
        self.assertEqual(
            len(self.dual_graph_filter_dicts["GA"].powers_of_A),
            self.dual_graph_filter_dicts["GA"].params["order"],
        )

        self.dual_graph_filter_dicts["GD"].precompute_polynomial()
        self.assertEqual(
            len(self.dual_graph_filter_dicts["GD"].powers_of_D),
            self.dual_graph_filter_dicts["GD"].params["order"],
        )

    def test_vandermonde_matrix(self):
        """
        Test Vandermonde matrix construction.
        """
        vdm = self.graph_filter.vandermonde_matrix(self.graph.operator.V, self.order)
        self.assertEqual(vdm.shape, (len(self.graph.operator.V), self.order))

        for filter_type in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            if filter_type == "GAGD":
                dual_vdm = self.dual_graph_filter_dicts[
                    filter_type
                ].vandermonde_matrix_compose(self.order)
                self.assertEqual(
                    dual_vdm.shape, (len(self.graph.operator.V), self.order * 2)
                )
            else:
                dual_vdm = self.dual_graph_filter_dicts[
                    filter_type
                ].vandermonde_matrix_compose(self.order)
                self.assertEqual(
                    dual_vdm.shape, (len(self.graph.operator.V), self.order)
                )

    def test_get_polynomial_coefficients(self):
        """
        Test retrieval of polynomial coefficients.
        """
        vdm, c = self.graph_filter.get_polynomial_coefficients(self.kernel, self.order)
        self.assertEqual(vdm.shape, (len(self.graph.operator.V), self.order))
        self.assertEqual(c.shape, (self.order,))

        for filter_type in ["GAGD", "GQAD", "GQDA", "GA", "GD"]:
            vdm, c = self.dual_graph_filter_dicts[
                filter_type
            ].get_polynomial_coefficients(self.kernel, self.order)
            if filter_type == "GAGD":
                self.assertEqual(
                    vdm.shape, (len(self.graph.operator.V), self.order * 2)
                )
                self.assertEqual(c.shape, (self.order * 2,))
            else:
                self.assertEqual(vdm.shape, (len(self.graph.operator.V), self.order))
                self.assertEqual(c.shape, (self.order,))

        # Instantiate only to verify no errors in computation
        # TODO: Add assertions to verify correctness of coefficients
        self.graph_filter.get_polynomial_coefficients(self.kernel, deg=-1)
        self.dual_graph_filter_dicts["GAGD"].get_polynomial_coefficients(
            self.kernel, deg=-1
        )


if __name__ == "__main__":
    unittest.main()
