"""Collision regressions without ROS or the external kinematics cache."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely import LineString, Point, STRtree


@pytest.fixture
def planner():
    # Load the actual numerical definitions without executing ROS imports or
    # reading the machine-specific pickle at module import time.
    source = Path(__file__).parents[1] / 'RRT' / 'RRT.py'
    module = ast.parse(source.read_text())
    definitions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in ('forward_kinematics_batch', 'RRTAngleBatch')
    ]
    namespace = {
        'np': np, 'LineString': LineString, 'STRtree': STRtree,
        'lengths': np.array([0.325, 0.275, 0.25]),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'),
         namespace)
    instance = namespace['RRTAngleBatch'].__new__(namespace['RRTAngleBatch'])
    instance.length = namespace['lengths']
    instance.obstacle_tree = STRtree([])
    return instance


def test_batch_mask_uses_configuration_indices(planner):
    planner.obstacle_tree = STRtree([
        Point(5, 5), Point(6, 6), Point(0.2, 0.01), Point(0.4, 0.01),
    ])
    result = planner.is_in_obstacle_batch([
        [0, 0, 0], [np.pi / 2, 0, 0], [0, 0, 0],
    ])
    np.testing.assert_array_equal(result, [True, False, True])
    assert result.dtype == np.bool_


@pytest.mark.parametrize('angles', [[], np.empty((0, 3))])
def test_empty_batch(planner, angles):
    assert planner.is_in_obstacle_batch(angles).shape == (0,)


def test_no_obstacles_and_single_configuration(planner):
    np.testing.assert_array_equal(
        planner.is_in_obstacle_batch([0, 0, 0]), [False])


@pytest.mark.parametrize('distance, expected', [
    (0.04999, True), (0.05, True), (0.05001, False),
])
def test_clearance_boundary(planner, distance, expected):
    planner.obstacle_tree = STRtree([Point(0.2, distance)])
    assert planner.is_in_obstacle_batch([[0, 0, 0]])[0] == expected


def test_round_cap_clearance_between_buffer_vertices(planner):
    theta = np.pi - np.pi / 64
    obstacle = Point(0.04999 * np.cos(theta), 0.04999 * np.sin(theta))
    # This point lies inside the true round cap but outside its old polygon.
    assert not LineString([(0, 0), (0.85, 0)]).buffer(0.05).intersects(obstacle)
    planner.obstacle_tree = STRtree([obstacle])
    assert planner.is_in_obstacle_batch([[0, 0, 0]])[0]


def test_obstacle_clearance_is_in_addition_to_robot_clearance(planner):
    obstacle = LineString([(0.2, 0.09), (0.4, 0.09)]).buffer(0.05)
    robot_line = LineString([(0, 0), (0.85, 0)])
    assert not robot_line.intersects(obstacle)
    planner.obstacle_tree = STRtree([obstacle])
    assert planner.is_in_obstacle_batch([[0, 0, 0]])[0]


@pytest.mark.parametrize('angles', [
    [0, 0], [[0, 0]], np.zeros((1, 1, 3)), [[np.nan, 0, 0]],
    [[np.inf, 0, 0]],
])
def test_invalid_angles(planner, angles):
    with pytest.raises(ValueError):
        planner.is_in_obstacle_batch(angles)


def test_random_batches_match_direct_distance(planner):
    rng = np.random.default_rng(42)
    obstacles = [Point(xy) for xy in rng.uniform(-1, 1, (30, 2))]
    planner.obstacle_tree = STRtree(obstacles)
    angles = rng.uniform(-np.pi, np.pi, (100, 3))
    expected = []
    for configuration in angles:
        line = LineString(planner.forward_kinematics(configuration).T)
        expected.append(any(line.distance(obstacle) <= 0.05 for obstacle in obstacles))
    np.testing.assert_array_equal(planner.is_in_obstacle_batch(angles), expected)


@pytest.mark.parametrize('failure', ['no_goal', 'start_collision', 'iterations_exhausted'])
def test_failed_planning_has_no_goal_error(planner, failure):
    from scipy.spatial import KDTree

    namespace = planner.build_tree.__globals__
    namespace['KDTree'] = KDTree
    namespace['kd_tree_kineamtics'] = SimpleNamespace(
        query_ball_point=lambda *_: [] if failure == 'no_goal' else [0])
    namespace['random_angles'] = np.array([[0, 0, 0]])
    namespace['tqdm'] = lambda iterable: iterable
    obstacles = [Point(0.2, 0)] if failure == 'start_collision' else []
    instance = type(planner)([0, 0, 0], obstacles)
    assert instance.error is None
    instance.error = 0.1  # A previous result must not leak into this attempt.
    assert instance.build_tree([0.85, 0], max_iterations=0) is None
    assert instance.error is None


def test_solver_returns_failure_without_reading_unavailable_error():
    source = Path(__file__).parents[1] / 'RRT' / 'RRT.py'
    module = ast.parse(source.read_text())
    definition = next(node for node in module.body
                      if isinstance(node, ast.ClassDef) and node.name == 'RRT')
    namespace = {
        'Node': object, 'np': np, 'LineString': LineString,
        'RRTAngleBatch': lambda *_: SimpleNamespace(build_tree=lambda *_: None),
    }
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'),
         namespace)
    node = namespace['RRT'].__new__(namespace['RRT'])
    warnings = []
    node.get_logger = lambda: SimpleNamespace(info=lambda _: None, warn=warnings.append)
    request = SimpleNamespace(obstacles=[], target=[0.85, 0], angles=[0, 0, 0])
    response = SimpleNamespace()
    assert node.solve_rrt(request, response) is response
    assert response.sucess is False
    assert response.angles == []
    assert warnings == ['Path planning failed: no path found']
