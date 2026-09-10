import math
import pytest
from solverpilot.applications.routing import *


def test_time_window_accepts_zero_length():
    w=VRPTimeWindow(2,2); assert w.start==w.end==2

@pytest.mark.parametrize('a,b',[(-1,2),(3,2),(math.nan,2),(1,math.inf),(True,2)])
def test_time_window_rejects_invalid(a,b):
    with pytest.raises(VRPValidationError): VRPTimeWindow(a,b)


def test_node_ids_unique():
    with pytest.raises(VRPValidationError):
        VRPInstance([VRPNode('D'),VRPNode('D')],[VRPCustomer('A','D')],[VRPVehicle('v','D')],[[0,0],[0,0]])


def test_customer_ids_unique(cvrp_small):
    with pytest.raises(VRPValidationError):
        VRPInstance(cvrp_small.nodes,[VRPCustomer('x','A'),VRPCustomer('x','B')],cvrp_small.vehicles,cvrp_small.distances)


def test_vehicle_ids_unique(cvrp_small):
    with pytest.raises(VRPValidationError):
        VRPInstance(cvrp_small.nodes,cvrp_small.customers,[VRPVehicle('v','D'),VRPVehicle('v','D')],cvrp_small.distances)


def test_customer_unknown_node_rejected(cvrp_small):
    with pytest.raises(VRPValidationError):
        VRPInstance(cvrp_small.nodes,[VRPCustomer('x','ZZ')],cvrp_small.vehicles,cvrp_small.distances)


def test_vehicle_unknown_node_rejected(cvrp_small):
    with pytest.raises(VRPValidationError):
        VRPInstance(cvrp_small.nodes,cvrp_small.customers,[VRPVehicle('v','ZZ')],cvrp_small.distances)


def test_multiple_jobs_may_share_one_location(cvrp_small):
    inst=VRPInstance(cvrp_small.nodes,[VRPCustomer('x','A'),VRPCustomer('y','A')],cvrp_small.vehicles,cvrp_small.distances)
    assert tuple(c.node_id for c in inst.customers)==('A','A')

@pytest.mark.parametrize('bad',[
    [[0,1],[1]], [[0,-1],[-1,0]], [[0,float('nan')],[1,0]], [[0,True],[1,0]],
])
def test_matrix_rejects_invalid(bad):
    with pytest.raises(VRPValidationError): VRPMatrix(bad)


def test_matrix_size_must_match_nodes(cvrp_small):
    with pytest.raises(VRPValidationError):
        VRPInstance(cvrp_small.nodes,cvrp_small.customers,cvrp_small.vehicles,[[0,1],[1,0]])


def test_time_constraints_require_explicit_travel_time():
    nodes=[VRPNode('D'),VRPNode('A')]
    with pytest.raises(VRPValidationError, match='explicit travel-time'):
        VRPInstance(nodes,[VRPCustomer('A','A',time_window=VRPTimeWindow(0,2))],[VRPVehicle('v','D')],[[0,1],[1,0]])


def test_service_duration_alone_does_not_force_time_matrix():
    nodes=[VRPNode('D'),VRPNode('A')]
    x=VRPInstance(nodes,[VRPCustomer('A','A',service_duration=5)],[VRPVehicle('v','D')],[[0,1],[1,0]])
    assert not x.has_time_constraints


def test_vehicle_capacity_positive():
    with pytest.raises(VRPValidationError): VRPVehicle('v','D',capacity=0)


def test_customer_values_nonnegative():
    with pytest.raises(VRPValidationError): VRPCustomer('x','A',demand=-1)
    with pytest.raises(VRPValidationError): VRPCustomer('x','A',service_duration=-1)


def test_bool_not_silently_numeric():
    with pytest.raises(VRPValidationError): VRPCustomer('x','A',demand=True)
    with pytest.raises(VRPValidationError): VRPVehicle('v','D',capacity=True)


def test_instance_maps_are_read_only(cvrp_small):
    with pytest.raises(TypeError): cvrp_small.node_index['x']=1
    with pytest.raises(TypeError): cvrp_small.customer_map['x']=cvrp_small.customers[0]


def test_distance_and_time_are_separate(vrptw_small):
    assert vrptw_small.distance('D','A')==1
    assert vrptw_small.travel_time('D','A')==2


def test_top_level_api_freeze_preserved():
    import solverpilot
    assert len(solverpilot.__all__)==78
    assert 'VRPInstance' not in solverpilot.__all__


def test_multiple_customers_may_share_one_physical_node():
    nodes=[VRPNode('D'),VRPNode('L')]
    inst=VRPInstance(nodes,[VRPCustomer('a','L'),VRPCustomer('b','L')],[VRPVehicle('v','D',capacity=2)],[[0,1],[1,0]])
    assert len(inst.customers)==2


def test_route_distance_rejects_unknown_references(cvrp_small):
    with pytest.raises(VRPValidationError): route_distance(cvrp_small,VRPRoute('bad',['A']))
    with pytest.raises(VRPValidationError): route_distance(cvrp_small,VRPRoute('v1',['bad']))
