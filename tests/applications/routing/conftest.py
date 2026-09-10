import pytest
from solverpilot.applications.routing import VRPCustomer, VRPInstance, VRPNode, VRPTimeWindow, VRPVehicle


@pytest.fixture
def cvrp_small():
    nodes=[VRPNode('D'),VRPNode('A'),VRPNode('B'),VRPNode('C')]
    d=[[0,2,3,4],[2,0,2,5],[3,2,0,2],[4,5,2,0]]
    return VRPInstance(
        nodes,
        [VRPCustomer('A','A',2),VRPCustomer('B','B',2),VRPCustomer('C','C',2)],
        [VRPVehicle('v1','D',capacity=4),VRPVehicle('v2','D',capacity=4)],
        d,
        name='small',
    )


@pytest.fixture
def vrptw_small():
    nodes=[VRPNode('D'),VRPNode('A'),VRPNode('B')]
    d=[[0,1,2],[1,0,1],[2,1,0]]
    t=[[0,2,4],[2,0,2],[4,2,0]]
    return VRPInstance(
        nodes,
        [
            VRPCustomer('A','A',1,1,VRPTimeWindow(2,5)),
            VRPCustomer('B','B',1,1,VRPTimeWindow(5,9)),
        ],
        [VRPVehicle('v','D',capacity=2,shift_window=VRPTimeWindow(0,12),max_route_duration=12)],
        d,
        travel_times=t,
        name='tw',
    )
