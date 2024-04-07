import sys

import numpy as np

import py_pore_flow
from py_pore_flow.simulation.simulation_base import import_network
from py_pore_flow.network import Network
from py_pore_flow.simulation.simulation_base import KrelCalculator
from py_pore_flow.simulation.imbibition import ImbibitionSimulation


class my_class:
    def __init__(self):
        np.set_printoptions(threshold=sys.maxsize)

        pn_before = import_network("C:\\ltrace\\GeoSlicer-2.1.0-2023-10-24-win-amd64\\LTrace\\temp\\__SlicerTemp__2024-04-05_10+46+12.984\\fqsvCkPplyiROvWQsaxiZq\\pn_glitch_before.bin")
        pn_after = import_network("C:\\ltrace\\GeoSlicer-2.1.0-2023-10-24-win-amd64\\LTrace\\temp\\__SlicerTemp__2024-04-05_10+46+12.984\\fqsvCkPplyiROvWQsaxiZq\\pn_glitch_after.bin")

        self.network_b = Network(pn_before)
        imbibition_b = ImbibitionSimulation(self.network_b, 40000)
        imbibition_b.update_pc(1285.92)
        Sw = imbibition_b.calculate_sw_total()
        Krw, Kro = imbibition_b.calculate_krel()
        print("Before: Sw:", Sw, "Krw:", Krw, "Kro:", Kro)

        self.network_a = Network(pn_after)
        imbibition_a = ImbibitionSimulation(self.network_a, 40000)
        imbibition_a.update_pc(1284.75)
        Sw = imbibition_a.calculate_sw_total()
        Krw, Kro = imbibition_a.calculate_krel()
        print("After: Sw:", Sw, "Krw:", Krw, "Kro:", Kro)

        # pn_initial_p = import_network("SKvCUYjHtlQIEOkrKyCzpl/pn_initial.bin")
        # pn_initial_n = import_network("SKvCUYjHtlQIEOkrKyCzpl/pn_initial.bin")
        # # pn_original = import_network("SKvCUYjHtlQIEOkrKyCzpl/pn_original.bin")
        # pn_positive_before = import_network("SKvCUYjHtlQIEOkrKyCzpl/pn_positive_before.bin")
        # pn_negative_before = import_network("SKvCUYjHtlQIEOkrKyCzpl/pn_negative_before.bin")

        # self.network_p = Network(pn_positive_before)
        # imbibition_p = ImbibitionSimulation(self.network_p, 40000)
        # pore_saturation_before = pn_positive_before["pore.water_saturation"]
        # throat_saturation_before = pn_positive_before["throat.water_saturation"]
        # # print("Sw_p before:", pn_positive_before["pore.water_saturation"].sum() + pn_positive_before["throat.water_saturation"].sum())
        # print(
        #     "oilayer before p:",
        #     (pn_positive_before["pore.corner_oil_layer"]).sum() + (pn_positive_before["throat.corner_oil_layer"]).sum(),
        # )
        # print(
        #     "oilfilled before p:",
        #     (pn_positive_before["pore.oil_filled"]).sum() + (pn_positive_before["throat.oil_filled"]).sum(),
        # )
        # imbibition_p.update_pc(137.40)
        # pore_saturation_after = pn_positive_before["pore.water_saturation"]
        # throat_saturation_after = pn_positive_before["throat.water_saturation"]
        # Sw = imbibition_p.calculate_sw_total()
        # # print("Sw_p after:", pn_positive_before["pore.water_saturation"].sum() + pn_positive_before["throat.water_saturation"].sum())
        # Krw, Kro = imbibition_p.calculate_krel()
        # # print("Sw_p:", Sw)
        # # print("Krw_p:", Krw)
        # print("How many pores p:", (pore_saturation_before > pore_saturation_after).sum())
        # print("How many throats p:", (throat_saturation_before > throat_saturation_after).sum())
        # print(
        #     "oilayer after p:",
        #     (pn_positive_before["pore.corner_oil_layer"]).sum() + (pn_positive_before["throat.corner_oil_layer"]).sum(),
        # )
        # print(
        #     "oilfilled after p:",
        #     (pn_positive_before["pore.oil_filled"]).sum() + (pn_positive_before["throat.oil_filled"]).sum(),
        # )
        # print()

        # self.network_n = Network(pn_negative_before)
        # imbibition_n = ImbibitionSimulation(self.network_n, 40000)
        # pore_saturation_before = pn_negative_before["pore.water_saturation"]
        # throat_saturation_before = pn_negative_before["throat.water_saturation"]
        # # print("SwSum_n before:", pn_negative_before["pore.water_saturation"].sum() + pn_negative_before["throat.water_saturation"].sum())
        # # Sw = imbibition_n.calculate_sw_total()
        # # print("Sw total N before:", Sw)
        # print(
        #     "oilayer before n:",
        #     (pn_negative_before["pore.corner_oil_layer"]).sum() + (pn_negative_before["throat.corner_oil_layer"]).sum(),
        # )
        # print(
        #     "oilfilled before n:",
        #     (pn_negative_before["pore.oil_filled"]).sum() + (pn_negative_before["throat.oil_filled"]).sum(),
        # )
        # imbibition_n.update_pc(-15.34)
        # pore_saturation_after = pn_negative_before["pore.water_saturation"]
        # throat_saturation_after = pn_negative_before["throat.water_saturation"]
        # Sw = imbibition_n.calculate_sw_total()
        # # print("SwSum_n after:", pn_negative_before["pore.water_saturation"].sum() + pn_negative_before["throat.water_saturation"].sum())
        # Krw, Kro = imbibition_n.calculate_krel()
        # # print("Sw total N after:", Sw)
        # # print("Krw_n:", Krw)
        # print("How many pores n:", (pore_saturation_before > pore_saturation_after).sum())
        # print("How many throats n:", (throat_saturation_before > throat_saturation_after).sum())
        # print(
        #     "oilayer after n:",
        #     (pn_negative_before["pore.corner_oil_layer"]).sum() + (pn_negative_before["throat.corner_oil_layer"]).sum(),
        # )
        # print(
        #     "oilfilled after n:",
        #     (pn_negative_before["pore.oil_filled"]).sum() + (pn_negative_before["throat.oil_filled"]).sum(),
        # )
        # # print("oil saturation! n:", np.sort(pn_negative_before["pore.oil_saturation"][(pn_negative_before["pore.oil_filled"] == False)]))

        # # pn_negative = import_network("SKvCUYjHtlQIEOkrKyCzpl/pn_negative.bin")

        # # self.network_p = Network(pn_initial_p)
        # # self.krel_calc_p = KrelCalculator(self.network_p)
        # # self.network_p.pn.update(pn_positive.copy())
        # # print("KRW positive:", self.krel_calc_p.calculate_krw())

        # # self.network_n = Network(pn_initial_n)
        # # self.krel_calc_n = KrelCalculator(self.network_n)
        # # self.network_n.pn.update(pn_negative.copy())
        # # print("KRW negative:", self.krel_calc_n.calculate_krw())

        # # print()

        # # print("Positive cond:", self.krel_calc_p.store)
        # # print("Negative cond:", self.krel_calc_n.store)
        # # # print("GREATER SUM:", (self.krel_calc_p.store > self.krel_calc_n.store).sum())
        # # print("Len positive", len(self.krel_calc_p.store))
        # # print("Len negative", len(self.krel_calc_n.store))
        # # print("P pore sw: ", (pn_positive["pore.water_saturation"]).sum())
        # # print("P throat sw: ", (pn_positive["throat.water_saturation"]).sum())

        # # print("N pore sw: ", (pn_negative["pore.water_saturation"]).sum())
        # # print("N throat sw: ", (pn_negative["throat.water_saturation"]).sum())


teste = my_class()
